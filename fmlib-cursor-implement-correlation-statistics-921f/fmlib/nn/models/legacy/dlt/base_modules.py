import math

import torch
import torch.nn as nn
import torch.nn.functional as func


def _init_rope_parameters(head_dim, rope_theta=10000.0, partial_rotary_factor=1, device=None):
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
    return inv_freq


class RotaryEmbedding(nn.Module):
    def __init__(self, hid_dim, device=None):
        super().__init__()
        inv_freq = _init_rope_parameters(hid_dim, device=device)
        self.register_buffer("inv_freq", inv_freq)

    @torch.no_grad()
    def forward(self, positions, x):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(positions.shape[0], -1, 1)
        positions_expanded = positions[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float().to(x.device) @ positions_expanded.float()).transpose(1, 2)
            emb = torch.cat([freqs, freqs], dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(x.dtype), sin.to(x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, positions=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class AttentionModule(nn.Module):
    def __init__(self, n_embed, n_head, drop_rate=0.1):
        super().__init__()
        assert n_embed % n_head == 0
        self.q_layer = nn.Linear(n_embed, n_embed)
        self.k_layer = nn.Linear(n_embed, n_embed)
        self.v_layer = nn.Linear(n_embed, n_embed)
        self.proj_layer = nn.Linear(n_embed, n_embed)
        self.attn_dropout = nn.Dropout(drop_rate)
        self.resid_dropout = nn.Dropout(drop_rate)
        self.drop_rate = drop_rate
        self.n_head = n_head
        self.n_embed = n_embed

    def forward(self, x_0, x_1, attn_mask=None, rotary_position_embeds=None):
        batch, seq_0, dim = x_0.size()
        _, seq_1, _ = x_1.size()

        q = self.q_layer(x_0)
        k = self.k_layer(x_1)
        v = self.v_layer(x_1)

        q = q.view(batch, seq_0, self.n_head, dim // self.n_head).transpose(1, 2)
        k = k.view(batch, seq_1, self.n_head, dim // self.n_head).transpose(1, 2)
        v = v.view(batch, seq_1, self.n_head, dim // self.n_head).transpose(1, 2)

        if rotary_position_embeds is not None:
            cos, sin = rotary_position_embeds
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        att_score = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if attn_mask is not None:
            att_score = att_score + attn_mask

        att_score = func.softmax(att_score, dim=-1)
        att_score = self.attn_dropout(att_score)
        out = att_score @ v
        out = out.transpose(1, 2).contiguous().view(batch, seq_0, dim)

        out = self.resid_dropout(self.proj_layer(out))
        out = (out, att_score)

        return out


class EncoderBlock(nn.Module):
    def __init__(self, n_embed, n_head, drop_rate=0.1, use_rotary=False, **kwargs):
        super().__init__()
        self.ln_0 = nn.LayerNorm(n_embed)
        self.self_attn = AttentionModule(n_embed=n_embed, n_head=n_head, drop_rate=drop_rate)
        self.ln_1 = nn.LayerNorm(n_embed)
        self.mlp = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(drop_rate),
        )
        self.use_rotary = use_rotary
        if self.use_rotary:
            self.rotary_emb = RotaryEmbedding(n_embed // n_head)

    def forward(
        self,
        x,
        attn_mask=None,
        positions=None,
    ):
        _, seq_len, _ = x.shape
        if self.use_rotary:
            if positions is None:
                positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
            position_embeddings = self.rotary_emb(positions, x)
        else:
            position_embeddings = None

        inp_x = self.ln_0(x)
        att_output = self.self_attn(inp_x, inp_x, attn_mask=attn_mask, rotary_position_embeds=position_embeddings)
        x_self_att = att_output[0]
        att_score = att_output[1]

        x = x + x_self_att
        x = x + self.mlp(self.ln_1(x))
        out = (x, att_score)
        return out


class CrossBlock(nn.Module):
    def __init__(self, n_embed, n_head, drop_rate=0.1, use_rotary=False, **kwargs):
        super().__init__()
        self.ln_0 = nn.LayerNorm(n_embed)
        self.ln_1 = nn.LayerNorm(n_embed)
        self.cross_attn = AttentionModule(n_embed=n_embed, n_head=n_head, drop_rate=drop_rate)
        self.ln_2 = nn.LayerNorm(n_embed)
        self.mlp = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(drop_rate),
        )
        self.use_rotary = use_rotary
        if self.use_rotary:
            self.rotary_emb = RotaryEmbedding(n_embed // n_head)

    def forward(
        self,
        x,
        x_kv,
        attn_mask=None,
        positions=None,
    ):
        position_embeddings = None
        att_output = self.cross_attn(
            self.ln_0(x), self.ln_1(x_kv), attn_mask=attn_mask, rotary_position_embeds=position_embeddings
        )
        x_cross_att = att_output[0]
        att_score = att_output[1]

        x = x + x_cross_att
        x = x + self.mlp(self.ln_2(x))
        out = (x, att_score)
        return out
