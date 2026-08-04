"""Backend capability contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class BackendCapabilities:
    """Declarative capabilities of a dataframe backend.

    Args:
        name: Backend identifier (``spark``, ``pandas``, ``polars``).
        supports_distributed: Whether computation can stay distributed.
        requires_local_materialization: Whether data must be collected locally.
    """

    name: str
    supports_distributed: bool
    requires_local_materialization: bool


class BackendAdapter(Protocol):
    """Minimal protocol for dataframe adapters used by stages."""

    capabilities: BackendCapabilities

    def get_columns(self: BackendAdapter, data: object) -> list[str]:
        """Return column names for ``data``."""
        ...

    def project_columns(self: BackendAdapter, data: object, columns: list[str]) -> object:
        """Return a projection of ``data`` onto ``columns``."""
        ...
