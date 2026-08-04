from typing import TypedDict

import torch
import torch.nn as nn

from fmlib.constants.batches import GeneralBatch
from fmlib.nn.models.heads import DLTClassificationHead

from .base_model import BaseModelClass
from .local_models import CrossEncoder, EncoderBlock, SimpleDecoderModel


class BaseModelOutput(TypedDict):
    patches_mask: torch.BoolTensor
    encoded_input: torch.Tensor
    encoded_patches: torch.Tensor


class DLTmodelBase(BaseModelClass):
    def __init__(
        self,
        vocab=5504,
        hid_dim=512,
        num_heads=8,
        local_model_layers=3,
        main_model_layers=6,
        patch_size=24 * 60 * 60,
        local_attn_res=5 * 60,
        drop_rate=0.1,
        **kwargs,
    ):
        super().__init__()
        self.input_dropout = nn.Dropout(drop_rate)
        self.embedding_layer = nn.Embedding(vocab, hid_dim)
        self.pos_embedding_layer = nn.Embedding(384, hid_dim)
        self.cls_embed = nn.Parameter(torch.randn(1, 1, 1, hid_dim) / 10)
        self.patch_start_embed = nn.Parameter(torch.randn(1, 1, hid_dim) / 10)
        self.input_encoder = EncoderBlock(hid_dim, num_heads, drop_rate, use_rotary=False)
        self.local_cross_encoder = CrossEncoder(hid_dim, num_heads, local_model_layers)
        self.Model = SimpleDecoderModel(hid_dim, num_heads, main_model_layers, drop_rate, use_rotary=True)
        self.patch_size = patch_size
        self.local_attn_res = local_attn_res

        self.apply(self._init_weights)

    def forward(
        self,
        data: torch.Tensor,
        timestamps: torch.Tensor,
        patches: torch.Tensor,
        patches_pos: torch.Tensor,
        patches_mask: torch.Tensor,
        encoder_mask: torch.Tensor,
        encoder_cross_mask: torch.Tensor,
        decoder_cross_mask: torch.Tensor,
        **kwargs,
    ) -> GeneralBatch:
        pos_data = self.pos_embedding_layer(patches_pos).mean(-2)
        pos_data = self.input_dropout(pos_data)

        inp_data = self.embedding_layer(data)
        inp_data = torch.cat([self.cls_embed.repeat(inp_data.shape[0], inp_data.shape[1], 1, 1), inp_data], dim=2)
        inp_data = self.input_dropout(inp_data)
        dim_0, dim_1, dim_2, dim_3 = inp_data.shape
        inp_data = inp_data.view(-1, dim_2, dim_3)
        inp_data = self.input_encoder(inp_data)[0]
        inp_data = inp_data.view(dim_0, dim_1, dim_2, dim_3)
        inp_data = inp_data[:, :, 0, :]

        enc_mask = (1 - encoder_mask) * torch.finfo(inp_data.dtype).min
        enc_mask = enc_mask.unsqueeze(1)
        enc_mask = enc_mask.to(inp_data.dtype)

        enc_cross_mask = (1 - encoder_cross_mask) * torch.finfo(inp_data.dtype).min
        enc_cross_mask = enc_cross_mask.unsqueeze(1)
        enc_cross_mask = enc_cross_mask.to(inp_data.dtype)

        dec_cross_mask = (1 - decoder_cross_mask) * torch.finfo(inp_data.dtype).min
        dec_cross_mask = dec_cross_mask.unsqueeze(1)
        dec_cross_mask = dec_cross_mask.to(inp_data.dtype)

        encoder_positions = (timestamps % self.patch_size) // self.local_attn_res
        encoded_input, encoded_patches = self.local_cross_encoder(
            inp_data, pos_data, enc_mask, enc_cross_mask, encoder_positions
        )

        encoded_patches = torch.cat([self.patch_start_embed.repeat(encoded_patches.shape[0], 1, 1), encoded_patches], dim=1)
        patches = torch.cat([patches[:, :1], patches], dim=1)
        patches_mask = torch.cat([patches_mask[:, :1], patches_mask], dim=1)
        encoded_patches = self.Model(encoded_patches, patches)

        result: BaseModelOutput = BaseModelOutput(
            patches_mask=patches_mask, encoded_input=encoded_input, encoded_patches=encoded_patches
        )
        return result


class DLTmodelCLS(BaseModelClass):
    def __init__(
        self,
        vocab=5504,
        hid_dim=512,
        num_heads=8,
        local_model_layers=3,
        main_model_layers=6,
        patch_size=24 * 60 * 60,
        local_attn_res=5 * 60,
        output_layers=None,
        drop_rate=0.1,
        **kwargs,
    ):
        if output_layers is None:
            output_layers = ["dummy"]
        super().__init__()
        self.base_model = DLTmodelBase(
            vocab=vocab,
            hid_dim=hid_dim,
            num_heads=num_heads,
            local_model_layers=local_model_layers,
            main_model_layers=main_model_layers,
            patch_size=patch_size,
            local_attn_res=local_attn_res,
            drop_rate=drop_rate,
        )

        self.output_layers = nn.ModuleDict()
        for name in output_layers:
            self.output_layers[name] = DLTClassificationHead(hid_dim, 2)

        self.apply(self._init_weights)

    def forward(
        self,
        data: torch.Tensor,
        timestamps: torch.Tensor,
        patches: torch.Tensor,
        patches_pos: torch.Tensor,
        patches_mask: torch.Tensor,
        encoder_mask: torch.Tensor,
        encoder_cross_mask: torch.Tensor,
        decoder_cross_mask: torch.Tensor,
        **kwargs,
    ) -> GeneralBatch:
        base_result = self.base_model(
            data=data,
            timestamps=timestamps,
            patches=patches,
            patches_pos=patches_pos,
            patches_mask=patches_mask,
            encoder_mask=encoder_mask,
            encoder_cross_mask=encoder_cross_mask,
            decoder_cross_mask=decoder_cross_mask,
            **kwargs,
        )

        encoded_patches = base_result["encoded_patches"]
        out_pos = base_result["patches_mask"].long().argmin(-1) - 1
        arange = torch.arange(encoded_patches.shape[0], device=out_pos.device)
        encoded_patches = encoded_patches[arange, out_pos]
        result = {"encoded_patches": encoded_patches}
        for key in sorted(self.output_layers.keys()):
            result[key] = self.output_layers[key](encoded_patches)

        return result
