from typing import Any

from src.domain.vector import VectorDBConfig
from src.infrastructure.vector.base import BaseVectorDatabase


class FakePydanticConfig:
    """Minimal pydantic-like object exposing model_dump()."""

    def __init__(self, **values: Any):
        self._values = values

    def model_dump(self) -> dict[str, Any]:
        return dict(self._values)


class DummyClient:
    """Small client double used for constructor assertions."""

    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs

    def close(self) -> None:
        return None


__all__ = [
    "BaseVectorDatabase",
    "DummyClient",
    "FakePydanticConfig",
    "VectorDBConfig",
]
