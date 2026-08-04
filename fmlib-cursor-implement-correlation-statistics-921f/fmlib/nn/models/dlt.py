from typing import Self, TypedDict

import torch

from fmlib.constants.batches import GeneralBatch
from fmlib.nn.blocks import (
    CrossEncoderBlock,
    DropoutLastPositionWiseFFN,
    RotaryCrossEncoderModel,
    RotaryDecoderModel,
    RotaryEncoderBlock,
)
from fmlib.nn.models.heads import DLTClassificationHead
from fmlib.nn.utils.initialization import clone_module_to_moduledict, nba_init_weights
from fmlib.utils.deprecation import deprecation_warning


class BaseModelOutput(TypedDict):
    """
    Класс выводов базовой модели DLT.

    Может использоваться как для классификации, так и генерации (в перспективе).
    """

    patches_mask: torch.Tensor
    encoded_input: torch.Tensor
    encoded_patches: torch.Tensor


def _dlt_patches_pooling(encoded_patches: torch.Tensor, patches_mask: torch.Tensor) -> torch.Tensor:
    """Получает эмбеддинги первых значимых патчей."""
    device: torch.device = encoded_patches.device
    out_pos: torch.Tensor = patches_mask.long().argmin(-1) - 1
    arange: torch.Tensor = torch.arange(encoded_patches.shape[0], device=device)
    result: torch.Tensor = encoded_patches[arange, out_pos]
    return result


class DLTBody(torch.nn.Module):
    """
    Базовая модель DLT.

    Аргументы:
        hidden_dim (int): размерность скрытого слоя.
        patch_size (int): размер патча. Время в секундах.
        local_attn_res (int): разрешение локального attention. Время в секундах.
        embedding_layer (torch.nn.Module): слой эмбеддингов. Как правило `torch.nn.Embedding`.
        pos_embedding_layer (torch.nn.Module): слой эмбеддингов позиций. Как правило `torch.nn.Embedding`.
        input_encoder (torch.nn.Module): слой энкодера. Как правило `RotaryEncoderBlock`.
        local_cross_encoder (torch.nn.Module): слой локального кросс-энкодера. Как правило `CrossEncoderBlock`.
        patches_decoder (torch.nn.Module): слой декодера патчей. Как правило `RotaryDecoderModel`.
        dropout (float): dropout в модели. Используется на входе.

    Аттрибуты:
        hidden_dim (int): размерность скрытого слоя.
        patch_size (int): размер патча. Время в секундах.
        local_attn_res (int): разрешение локального attention. Время в секундах.
        input_encoder (torch.nn.Module): слой энкодера.
        local_cross_encoder (torch.nn.Module): слой локального кросс-энкодера.
        patches_decoder (torch.nn.Module): слой декодера.
        cls_embed (torch.nn.Parameter): обучаемый эмбеддинг для классификации.
            Имеет размер `1 x 1 x 1 x hidden_dim`.
        patch_start_embed (torch.nn.Parameter): обучаемый эмбеддинг для начала патчей.
            Имеет размер `1 x 1 x hidden_dim`.
        input_dropout (torch.nn.Dropout): dropout на входе. По умолчанию - 0.1.
    """

    def __init__(
        self: Self,
        hidden_dim: int,
        patch_size: int,
        local_attn_res: int,
        embedding_layer: torch.nn.Module,
        pos_embedding_layer: torch.nn.Module,
        input_encoder: torch.nn.Module,
        local_cross_encoder: torch.nn.Module,
        patches_decoder: torch.nn.Module,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.hidden_dim: int = hidden_dim
        self.patch_size: int = patch_size
        self.local_attn_res: int = local_attn_res

        self.input_dropout: torch.nn.Dropout = torch.nn.Dropout(dropout)

        self.embedding_layer: torch.nn.Module = embedding_layer
        self.pos_embedding_layer: torch.nn.Module = pos_embedding_layer

        self.input_encoder: torch.nn.Module = input_encoder
        self.local_cross_encoder: torch.nn.Module = local_cross_encoder
        self.patches_decoder: torch.nn.Module = patches_decoder

        self.cls_embed: torch.nn.Parameter = torch.nn.Parameter(0.1 * torch.randn(1, 1, 1, hidden_dim))
        self.patch_start_embed: torch.nn.Parameter = torch.nn.Parameter(0.1 * torch.randn(1, 1, hidden_dim))

        self.apply(nba_init_weights)

    def forward(
        self: Self,
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
        pos_data: torch.Tensor = self.pos_embedding_layer(patches_pos).mean(-2)
        pos_data = self.input_dropout(pos_data)

        inp_data: torch.Tensor = self.embedding_layer(data)
        batch_size, seq_len, _, _ = inp_data.shape
        cls_embed: torch.Tensor = self.cls_embed.repeat(batch_size, seq_len, 1, 1)
        inp_data = torch.cat([cls_embed, inp_data], dim=2)
        inp_data = self.input_dropout(inp_data)

        dim_0, dim_1, dim_2, dim_3 = inp_data.size()
        inp_data = inp_data.view(-1, dim_2, dim_3)
        inp_data, _ = self.input_encoder(inp_data)
        inp_data = inp_data.view(dim_0, dim_1, dim_2, dim_3)
        inp_data = inp_data[:, :, 0, :]

        enc_mask: torch.BoolTensor = encoder_mask.unsqueeze(1).bool()
        enc_cross_mask: torch.BoolTensor = encoder_cross_mask.unsqueeze(1).bool()

        encoder_positions: torch.Tensor = timestamps % self.patch_size
        encoder_positions = encoder_positions // self.local_attn_res
        encoded_input, encoded_patches = self.local_cross_encoder(
            hidden_states=inp_data,
            patches=pos_data,
            attn_mask=enc_mask,
            cross_attn_mask=enc_cross_mask,
            positions=encoder_positions,
        )

        batch_size, _, _ = encoded_patches.shape
        patch_start_embed: torch.Tensor = self.patch_start_embed.repeat(batch_size, 1, 1)
        encoded_patches = torch.cat([patch_start_embed, encoded_patches], dim=1)
        patches = torch.cat([patches[:, :1], patches], dim=1)
        patches_mask = torch.cat([patches_mask[:, :1], patches_mask], dim=1)
        encoded_patches = self.patches_decoder(hidden_states=encoded_patches, positions=patches)

        result: BaseModelOutput = BaseModelOutput(
            patches_mask=patches_mask, encoded_input=encoded_input, encoded_patches=encoded_patches
        )

        return result


class DLTEmbedding(torch.nn.Module):
    """
    Модель получения эмбеддингов из тела DLT.

    Аргументы:
        base_model (torch.nn.Module): базовая модель. Как правило `DLTBody`.
    """

    def __init__(
        self,
        base_model: torch.nn.Module,
    ) -> None:
        super().__init__()

        self.base_model: torch.nn.Module = base_model
        self.apply(nba_init_weights)

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
        base_result: BaseModelOutput = self.base_model(
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

        encoded_patches: torch.Tensor = _dlt_patches_pooling(
            encoded_patches=base_result["encoded_patches"],
            patches_mask=base_result["patches_mask"],
        )

        result = {"encoded_patches": encoded_patches.float()}

        return result


class DLTClassification(torch.nn.Module):
    """
    Модель классификации потребностей DLT.

    Аргументы:
        base_model (torch.nn.Module): базовая модель. Как правило `DLTBody`.
        output_layer_template (torch.nn.Module): шаблон слоя вывода. Как правило `DLTClassificationHead`.
        output_layers (list[str] | None): имена слоёв вывода. По умолчанию `["dummy"]`.
    """

    def __init__(
        self: Self,
        base_model: torch.nn.Module,
        output_layer_template: torch.nn.Module,
        output_layers: list[str] | None = None,
    ) -> None:
        if output_layers is None:
            output_layers = ["dummy"]
        super().__init__()

        if (actual_layers := len(output_layers)) != (unique_len := len(set(output_layers))):
            msg: str = f"Output layers must be unique. Got {unique_len=} unique vs. {actual_layers=} total."
            raise ValueError(msg)

        self.output_names: list[str] = sorted(output_layers)

        self.base_model: torch.nn.Module = base_model
        self.output_layers: torch.nn.ModuleDict = clone_module_to_moduledict(
            layer_names=self.output_names,
            module=output_layer_template,
            init="nba",
        )

        self.apply(nba_init_weights)

    def forward(
        self: Self,
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
        base_result: BaseModelOutput = self.base_model(
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

        encoded_patches: torch.Tensor = _dlt_patches_pooling(
            encoded_patches=base_result["encoded_patches"],
            patches_mask=base_result["patches_mask"],
        )

        result = {"encoded_patches": encoded_patches}
        for name in self.output_names:
            result[name] = self.output_layers[name](encoded_patches)

        return result


@deprecation_warning("Please use modular version of {class_name}.")
class LegacyDLTmodelCLS(DLTClassification):
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
        pos_vocab=384,
        **kwargs,
    ) -> None:
        if output_layers is None:
            output_layers = ["dummy"]

        embedding_layer: torch.nn.Embedding = torch.nn.Embedding(vocab, hid_dim)
        pos_embedding_layer: torch.nn.Embedding = torch.nn.Embedding(pos_vocab, hid_dim)

        ffn: DropoutLastPositionWiseFFN = DropoutLastPositionWiseFFN(
            linear_dim=(4 * hid_dim),
            model_dim=hid_dim,
            dropout=drop_rate,
        )

        input_encoder: RotaryEncoderBlock = RotaryEncoderBlock(
            num_heads=num_heads,
            hidden_dim=hid_dim,
            dropout=drop_rate,
            use_rotary=False,
            ffn=ffn,
        )

        encoder_layer_template: RotaryEncoderBlock = RotaryEncoderBlock(
            num_heads=num_heads,
            hidden_dim=hid_dim,
            dropout=drop_rate,
            use_rotary=True,
            ffn=ffn,
        )

        cross_layer_template: CrossEncoderBlock = CrossEncoderBlock(
            num_heads=num_heads,
            hidden_dim=hid_dim,
            dropout=drop_rate,
            ffn=ffn,
        )

        local_cross_encoder: RotaryCrossEncoderModel = RotaryCrossEncoderModel(
            encoder_layer_template=encoder_layer_template,
            cross_layer_template=cross_layer_template,
            num_layers=local_model_layers,
        )

        decoder_block_template: RotaryEncoderBlock = RotaryEncoderBlock(
            num_heads=num_heads,
            hidden_dim=hid_dim,
            dropout=drop_rate,
            use_rotary=True,
            ffn=ffn,
        )

        patches_decoder: RotaryDecoderModel = RotaryDecoderModel(
            layer_template=decoder_block_template,
            num_layers=main_model_layers,
            hidden_dim=hid_dim,
            dropout=drop_rate,
        )

        base_model: DLTBody = DLTBody(
            hidden_dim=hid_dim,
            patch_size=patch_size,
            local_attn_res=local_attn_res,
            embedding_layer=embedding_layer,
            pos_embedding_layer=pos_embedding_layer,
            input_encoder=input_encoder,
            local_cross_encoder=local_cross_encoder,
            patches_decoder=patches_decoder,
            dropout=drop_rate,
        )

        output_template = DLTClassificationHead(hid_dim)

        super().__init__(
            base_model=base_model,
            output_layers=output_layers,
            output_layer_template=output_template,
        )
