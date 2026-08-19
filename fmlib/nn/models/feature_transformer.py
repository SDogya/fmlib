import warnings
from copy import deepcopy
from typing import Dict, List, Optional, Self, Tuple

import torch
from torch.export import Dim

from fmlib.nn.blocks import EncoderBlock
from fmlib.nn.embedding import BaseFeatureEmbedding
from fmlib.nn.models.heads import BaseHead
from fmlib.nn.utils.agg import BaseAggregation
from fmlib.nn.utils.initialization import xavier_initialization
from fmlib.utils.named_adapters import OutputAdapter, validate_output_names


class FeatureTransformer(torch.nn.Module):
    """
    Модель для решения задачи классификации с использованием табличных и временных данных.

    Эта модель объединяет эмбеддинги категориальных и числовых признаков, использует несколько
    трансформерных блоков для обработки последовательностей, затем агрегирует результаты
    и передаёт в голову классификации. Также поддерживает позднее слияние (late fusion) скрытых состояний.
    Подразумевается, что раннее слияние (early fusion) может быть реализовано в передаваемом слое эмбеддингов (embedding).

    Args:
        num_encoder_layers (int): Количество трансформерных блоков.
        encoder_block (EncoderBlock): Блок трансформера, который будет повторён `num_encoder_layers` раз.
        aggregation_layer (BaseAggregation): Слой агрегации выходных эмбеддингов.
        head (BaseHead): Голова модели для классификации.
        embedding (Optional[BaseFeatureEmbedding]): Необязательный слой эмбеддингов для категориальных и числовых признаков.
        late_fusion_layer_norm (Optional[torch.nn.Module]): Экземпляр необязательного слоя нормализации
            для позднего слияния внешних скрытых состояний.
        output_names (Tuple[str, ...]): Имена выходов модели. По умолчанию — ("logits",).

    Attributes:
        embedding (BaseFeatureEmbedding): Слой эмбеддингов.
        transformer_blocks (torch.nn.ModuleList): Список трансформерных блоков.
        agg_layer (BaseAggregation): Слой агрегации.
        late_fusion_layer_norm (Optional[torch.nn.Module]): Слой нормализации для позднего слияния.
        head (BaseHead): Голова модели для классификации.
        output_adapter (OutputAdapter): Адаптер для формирования выходного словаря.
    """

    expected_output_count: int = 1

    def __init__(
        self: Self,
        num_encoder_layers: int,
        encoder_block: EncoderBlock,
        aggregation_layer: BaseAggregation,
        head: BaseHead,
        embedding: Optional[BaseFeatureEmbedding] = None,
        late_fusion_layer_norm: Optional[torch.nn.Module] = None,
        output_names: Tuple[str, ...] = ("logits",),
    ):
        """
        Инициализирует модель для задачи классификации.
        Все веса модели инициализируются с помощью `torch.nn.init.xavier_normal_`.

        Args:
            num_encoder_layers (int): Количество трансформерных блоков.
            encoder_block (EncoderBlock): Блок трансформера.
            aggregation_layer (BaseAggregation): Слой агрегации.
            head (torch.nn.Module): Голова классификации.
            embedding (Optional[BaseFeatureEmbedding]): Опциональный слой эмбеддингов.
                Слой можно не указывать, если вы собираетесь передавать в модель эмбеддинги собственной сборки.
            late_fusion_layer_norm (Optional[torch.nn.Module]): Слой нормализации для позднего слияния.
            output_names (Tuple[str, ...]): Имена выходов модели.
        """
        super().__init__()
        self.embedding = embedding
        self.transformer_blocks = torch.nn.ModuleList([deepcopy(encoder_block) for _ in range(num_encoder_layers)])
        self.agg_layer = aggregation_layer
        self.late_fusion_layer_norm = late_fusion_layer_norm
        self.head = head

        output_names = validate_output_names(output_names, self.expected_output_count)
        self.output_adapter: OutputAdapter = OutputAdapter(output_names)
        xavier_initialization(self)

    def get_input_names(self: Self) -> List[str]:
        """
        Возвращает список имён входных тензоров модели.
        Метод необходим для конвертации в ONNX.

        Returns:
            List[str]: Список имён входов.
        """
        return list(self.get_dynamic_shapes().keys())

    def get_output_names(self: Self) -> List[str]:
        """
        Возвращает список имён выходных тензоров модели.
        Метод необходим для конвертации в ONNX.

        Returns:
            List[str]: Список имён выходов.
        """
        return self.output_adapter.get_output_names()

    def get_dynamic_shapes(self: Self) -> Dict[str, Dict[int, str]]:
        """
        Возвращает информацию о динамических размерностях входных тензоров.
        Метод необходим для конвертации в ONNX.

        Returns:
            Dict[str, Dict[int, str]]: Словарь с описанием размерностей.
        """
        batch_dim = Dim.DYNAMIC
        if self.embedding is not None:
            return {
                "cat_features": {0: batch_dim, 1: Dim.STATIC},
                "num_features": {0: batch_dim, 1: Dim.STATIC},
                "hidden_states": {0: batch_dim, 1: Dim.STATIC},
            }
        shapes = {
            "input_embeddings": {0: batch_dim, 1: Dim.STATIC, 2: Dim.STATIC},
        }
        if self.late_fusion_layer_norm is not None:
            shapes["hidden_states"] = {0: batch_dim, 1: Dim.STATIC}
        return shapes

    def forward(
        self: Self,
        cat_features: Optional[torch.LongTensor] = None,
        num_features: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        input_embeddings: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Прямой проход через модель.

        Args:
            cat_features (torch.LongTensor): Тензор категориальных признаков размерности
                (batch_size, cat_feature_count).
            num_features (torch.Tensor): Тензор числовых признаков размерности
                (batch_size, num_feature_count).
            hidden_states (torch.Tensor): Тензор скрытых состояний размерности
                (batch_size, seq_state_len).
            input_embeddings (torch.Tensor): Тензор эмбеддингов размерности (batch_size, seq_len, embedding_dim).
                Опционально, вместо подачи cat_features и num_features вы можете напрямую передать эмбеддинги.
                Это полезно, если вы хотите получить больше контроля над преобразованием cat_features и num_features в
                связанные векторы, чем при использовании внутренней матрицы эмбедингов.
                Или при встраивании в более сложные иерархии классов.

        Returns:
            Dict[str, torch.Tensor]: Словарь с выходными значениями модели.

        Raises:
            ValueError: Если одновременно заданы `input_embeddings` и `cat_features/num_features`.
            ValueError: Если не заданы `input_embeddings` и отсутствует `embedding`.
            ValueError: Если `late_fusion_layer_norm` указан, но `hidden_states` не предоставлен.
        """
        if input_embeddings is not None and cat_features is not None and num_features is not None:
            msg: str = "You cannot specify both input_embeddings and cat_features/num_features"
            raise ValueError(msg)

        out: torch.Tensor
        if input_embeddings is None:
            if self.embedding is None:
                msg: str = "You must specify input_embeddings or embedding layer in __init__()"
                raise ValueError(msg)

            out = self.embedding(
                cat_features=cat_features,
                num_features=num_features,
                hidden_states=hidden_states,
            )
        else:
            if self.embedding is not None:
                warnings.warn("Cat_features/num_features is not given, but embedding layer are provided.", stacklevel=2)
            out = input_embeddings

        for transformer in self.transformer_blocks:
            out = transformer(out)

        attention_mask = torch.ones(*out.shape[:-1], device=out.device, dtype=torch.bool)
        aggregated_states: torch.Tensor = self.agg_layer(out, attention_mask)

        # late fusion branch: concat current hidden_states with input hidden_states
        if self.late_fusion_layer_norm is not None:
            if hidden_states is None:
                msg: str = "You must specify hidden_states for late fusion branch"
                raise ValueError(msg)

            hidden_states = self.late_fusion_layer_norm(hidden_states)
            aggregated_states = torch.cat([aggregated_states, hidden_states], dim=-1)

        output: torch.Tensor = self.head(aggregated_states)

        output_values: List[torch.Tensor] = [output]
        output_names: List[str] = self.get_output_names()
        return dict(zip(output_names, output_values, strict=False))


class UpliftFeatureTransformer(torch.nn.Module):
    """
    Модель для задач uplift моделирования, основанная на трансформер блоках.

    Эта модель использует общий механизм эмбеддинга для категориальных и числовых признаков,
    а также скрытых состояний любой другой модели. Входные данные обрабатываются дважды — один раз как воздействующая группа
    (treatment), другой — как контрольная (control), после чего вычисляются вероятности принадлежности
    к классу.

    Атрибуты:
        transformer (FeatureTransformer): Подмодель, реализующая логику классификации.
        embedding (BaseFeatureEmbedding): Слой для получения эмбеддингов из входных данных.
        treatment_features (torch.nn.Embedding): Эмбеддинг для кодирования группы воздействия.
        group_features (torch.nn.Embedding): Эмбеддинг для кодирования пользовательских групп.
    """

    expected_output_count: int = 2

    def __init__(
        self: Self,
        embedding: BaseFeatureEmbedding,
        num_encoder_layers: int,
        encoder_block: EncoderBlock,
        aggregation_layer: BaseAggregation,
        head: BaseHead,
        n_groups: int,
        late_fusion_layer_norm: Optional[torch.nn.Module] = None,
        output_names: Tuple[str, ...] = ("treatment_probs", "control_probs"),
    ):
        """
        Инициализация модели.
        Все веса модели инициализируются с помощью `torch.nn.init.xavier_normal_`.

        Args:
            embedding (BaseFeatureEmbedding): Слой эмбеддингов для категориальных и числовых признаков.
            num_encoder_layers (int): Количество трансформерных блоков.
            encoder_block (EncoderBlock): Блок трансформера, который будет использоваться.
            aggregation_layer (BaseAggregation): Слой агрегации эмбеддингов.
            head (BaseHead): Головной слой для классификации.
            n_groups (int): Количество пользовательских групп.
            late_fusion_layer_norm (Optional[torch.nn.Module]): Необязательный слой нормализации для позднего слияния.
            output_names (Tuple[str, ...]): Имена выходных тензоров модели.

        Raises:
            ValueError: Если количество имен выходов не равно 2.
        """
        super().__init__()
        self.transformer = FeatureTransformer(
            embedding=None,
            num_encoder_layers=num_encoder_layers,
            encoder_block=encoder_block,
            aggregation_layer=aggregation_layer,
            head=head,
            late_fusion_layer_norm=late_fusion_layer_norm,
            output_names=("logits",),
        )
        self.embedding = embedding
        self.treatment_features = torch.nn.Embedding(
            num_embeddings=2,
            embedding_dim=self.embedding.embedding_dim,
        )
        self.group_features = torch.nn.Embedding(
            num_embeddings=n_groups,
            embedding_dim=self.embedding.embedding_dim,
        )
        xavier_initialization(self)

        output_names = validate_output_names(output_names, self.expected_output_count)
        self.output_adapter: OutputAdapter = OutputAdapter(output_names)

    def get_input_names(self: Self) -> List[str]:
        """
        Возвращает список имён входных тензоров модели.
        Метод необходим для конвертации в ONNX.

        Returns:
            List[str]: Список имён входов.
        """
        return list(self.get_dynamic_shapes().keys())

    def get_output_names(self: Self) -> List[str]:
        """
        Возвращает список имён выходных тензоров модели.
        Метод необходим для конвертации в ONNX.

        Returns:
            List[str]: Список имён выходов.
        """
        return self.output_adapter.get_output_names()

    def get_dynamic_shapes(self: Self) -> Dict[str, Dict[int, str]]:
        """
        Возвращает информацию о динамических размерностях входных тензоров.
        Метод необходим для конвертации в ONNX.

        Returns:
            Dict[str, Dict[int, str]]: Словарь с описанием размерностей.
        """
        batch_dim = Dim.AUTO
        return {
            "cat_features": {0: batch_dim, 1: Dim.STATIC},
            "num_features": {0: batch_dim, 1: Dim.STATIC},
            "hidden_states": {0: batch_dim, 1: Dim.STATIC},
            "group": {0: batch_dim},
        }

    def forward(
        self: Self,
        cat_features: torch.LongTensor,
        num_features: torch.Tensor,
        hidden_states: torch.Tensor,
        group: torch.LongTensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Прямой проход через модель.

        Args:
            cat_features (torch.LongTensor): Тензор категориальных признаков.
            num_features (torch.Tensor): Тензор числовых признаков.
            hidden_states (torch.Tensor): Тензор временных состояний.
            group (torch.LongTensor): Тензор индексов групп наблюдения.

        Returns:
            Dict[str, torch.Tensor]: Словарь с вероятностями для воздействующей и контрольной групп.
        """
        out: torch.Tensor = self.embedding(
            cat_features=cat_features,
            num_features=num_features,
            hidden_states=hidden_states,
        )
        treat_embeds: torch.Tensor = self.treatment_features(torch.ones_like(group)).unsqueeze(1)
        control_embeds: torch.Tensor = self.treatment_features(torch.zeros_like(group)).unsqueeze(1)
        group_embeds: torch.Tensor = self.group_features(group).unsqueeze(1)
        treat_out: torch.Tensor = torch.cat([out, treat_embeds, group_embeds], dim=1)
        control_out: torch.Tensor = torch.cat([out, control_embeds, group_embeds], dim=1)

        treat_logits: torch.Tensor = self.transformer(
            hidden_states=hidden_states,
            input_embeddings=treat_out,
        )["logits"]
        control_logits: torch.Tensor = self.transformer(
            hidden_states=hidden_states,
            input_embeddings=control_out,
        )["logits"]

        t_probs: torch.Tensor = torch.softmax(treat_logits, dim=-1)[:, 1]
        c_probs: torch.Tensor = torch.softmax(control_logits, dim=-1)[:, 1]

        output_values: List[torch.Tensor] = [t_probs, c_probs]
        output_names: List[str] = self.get_output_names()
        return dict(zip(output_names, output_values, strict=False))
