from typing import Any, Dict, Optional, Self

import torch

from fmlib.nn.embedding.hidden_state_agg import BaseHiddenStateAggregator

from .base import BaseFeatureEmbedding
from .linear import LinearEmbedding


class FeatureEmbedding(BaseFeatureEmbedding):
    """
    Слой эмбеддингов для табличных данных.

    Объединяет категориальные и числовые признаки в единый эмбеддинг. Категориальные признаки
    обрабатываются с использованием `torch.nn.Embedding`, а числовые — линейным слоем. Также
    поддерживается добавление шума во время обучения и агрегация с внешними скрытыми состояниями.

    Args:
        numerical_feature_count (int): Количество числовых признаков.
        vocab_size (int): Размер словаря для категориальных признаков,
            включая специальные токены (например, 'unk').
        embedding_dim (int): Размерность эмбеддинга.
        std_noise (Optional[float]): Стандартное отклонение шума для добавления
            случайного шума во время обучения.
        nn_embedding_config (Dict[str, Any]): Дополнительные параметры для `torch.nn.Embedding`.
        hidden_state_aggregator (Optional[BaseHiddenStateAggregator]):
            Модуль для объединения табличных эмбеддингов с внешними скрытыми состояниями.

    Note:
        - Если есть только категориальные признаки, установите `numerical_feature_count = None`.
        - Если есть только числовые признаки, установите `vocab_size = None`.

    Example:
        >>> feature_emb = FeatureEmbedding(
        ...     numerical_feature_count=5,
        ...     vocab_size=1000,
        ...     embedding_dim=64,
        ...     std_noise=0.1
        ... )
        >>> cat_features = torch.randint(0, 1000, (32, 10))  # batch_size=32, n_cat_features=10
        >>> num_features = torch.rand(32, 5)  # batch_size=32, m_num_features=5
        >>> hidden_states = torch.rand(32, 64)
        >>> output = feature_emb(cat_features, num_features, hidden_states)
    """

    def __init__(
        self: Self,
        numerical_feature_count: int,
        vocab_size: int,
        embedding_dim: int,
        std_noise: Optional[float] = None,
        nn_embedding_config: Dict[str, Any] | None = None,
        hidden_state_aggregator: Optional[BaseHiddenStateAggregator] = None,
    ):
        """
        Инициализирует слой эмбеддингов для табличных данных.

        Args:
            numerical_feature_count (int): Количество числовых признаков.
            vocab_size (int): Размер словаря категориальных признаков.
            embedding_dim (int): Размерность эмбеддинга.
            std_noise (Optional[float]): Стандартное отклонение шума.
            nn_embedding_config (Dict[str, Any]): Дополнительные параметры для `torch.nn.Embedding`.
            hidden_state_aggregator (Optional[BaseHiddenStateAggregator]): Модуль для агрегации.
        """
        if nn_embedding_config is None:
            nn_embedding_config = {}
        super().__init__(embedding_dim=embedding_dim)
        self.num_embed = LinearEmbedding(
            feature_count=numerical_feature_count,
            embedding_dim=self.embedding_dim,
        )
        self.cat_embed = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            **nn_embedding_config,
        )
        self.std_noise = std_noise
        self.hidden_state_aggregator = hidden_state_aggregator

    def forward(
        self: Self,
        cat_features: torch.LongTensor,
        num_features: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Выполняет прямой проход через слой эмбеддингов.

        Args:
            cat_features (torch.LongTensor): Тензор категориальных признаков.
                Формат: (batch_size, n_cat_features).
            num_features (torch.Tensor): Тензор числовых признаков.
                Формат: (batch_size, m_num_features).
            hidden_states (torch.Tensor): Тензор скрытых состояний.
                Формат: (batch_size, embedding_dim).

        Returns:
            torch.Tensor: Объединённый эмбеддинг размерности (batch_size, (n_cat_features + m_num_features), emb_dim).
                Если используется `hidden_state_aggregator`,
                то размерность будет (batch_size, (n_cat_features + m_num_features + 1), emb_dim).
        """
        cat_embeds = self.cat_embed(cat_features)  # batch_size, n_cat_features, emb_dim
        num_embeds = self.num_embed(num_features)  # batch_size, m_num_features, emb_dim
        combined = torch.cat([cat_embeds, num_embeds], dim=1)

        if self.training and self.std_noise is not None:
            noise = torch.randn_like(combined) * self.std_noise
            combined = combined + noise
        if self.hidden_state_aggregator is not None:
            combined = self.hidden_state_aggregator(
                hidden_states=hidden_states,
                embeddings=combined,
            )
        return combined  # batch_size, (n_cat_features + m_num_features + 1 (if hidden_state_aggregator is not None)), emb_dim
