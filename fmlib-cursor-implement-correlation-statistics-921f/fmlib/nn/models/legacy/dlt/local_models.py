import torch
import torch.nn as nn

from .base_modules import CrossBlock, EncoderBlock


class CrossEncoder(nn.Module):
    def __init__(
        self,
        hid_dim=512,
        num_heads=8,
        num_layers=3,
        drop_rate=0.1,
        use_rotary=True,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.cross_attn_layers = nn.ModuleList()
        for _i in range(num_layers):
            self.layers.append(EncoderBlock(hid_dim, num_heads, drop_rate=drop_rate, use_rotary=use_rotary))
            self.cross_attn_layers.append(CrossBlock(hid_dim, num_heads, drop_rate=drop_rate, use_rotary=False))

    def forward(self, x, patches, attn_mask, cross_attn_mask, positions=None):
        encoded_x = x
        encoded_patches = patches
        for num, layer in enumerate(self.layers):
            encoded_x = layer(encoded_x, attn_mask, positions)[0]
            encoded_patches = self.cross_attn_layers[num](encoded_patches, encoded_x, attn_mask=cross_attn_mask)[0]
        return encoded_x, encoded_patches


class CrossDecoder(nn.Module):
    def __init__(
        self,
        hid_dim=512,
        num_heads=8,
        num_layers=3,
        drop_rate=0.1,
        use_rotary=True,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.cross_attn_layers = nn.ModuleList()
        for _i in range(num_layers):
            self.layers.append(EncoderBlock(hid_dim, num_heads, drop_rate=drop_rate, use_rotary=use_rotary))
            self.cross_attn_layers.append(CrossBlock(hid_dim, num_heads, drop_rate=drop_rate, use_rotary=False))

    def forward(self, x, patches, attn_mask, cross_attn_mask, positions=None):
        x.shape[1]
        encoded_x = x
        for num, layer in enumerate(self.layers):
            encoded_x = self.cross_attn_layers[num](encoded_x, patches, attn_mask=cross_attn_mask)[0]
            encoded_x = layer(encoded_x, attn_mask, positions)[0]
        return encoded_x


class SimpleDecoderModel(nn.Module):
    def __init__(
        self,
        hid_dim=512,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
        use_rotary=True,
        **kwargs,
    ):
        super().__init__()
        self.input_dropout = nn.Dropout(dropout)
        self.decoder = nn.ModuleList()
        for _ in range(num_layers):
            self.decoder.append(EncoderBlock(hid_dim, num_heads, dropout, use_rotary))
        self.ln_0 = nn.LayerNorm(hid_dim)

    def forward(self, x, positions=None):
        seq_len = x.shape[1]
        x = self.input_dropout(x)

        casual_mask = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len).to(x.device)
        casual_mask = (1 - casual_mask) * torch.finfo(x.dtype).min
        casual_mask = casual_mask.to(x.dtype)

        for layer in self.decoder:
            x = layer(x, attn_mask=casual_mask, positions=positions)[0]
        out = self.ln_0(x)
        return out


class OutputLayer(nn.Module):
    def __init__(self, hid_dim=512, output_dim=2):
        super().__init__()
        self.pad_tensor = nn.Parameter(torch.zeros(hid_dim))
        self.out_layer = nn.Linear(hid_dim, output_dim)

        torch.nn.init.normal_(self.out_layer.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.out_layer.bias)

    def forward(self, x):
        x = x + self.pad_tensor
        x = self.out_layer(x)
        return x
