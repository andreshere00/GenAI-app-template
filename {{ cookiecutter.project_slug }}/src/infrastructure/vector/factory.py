{%- if cookiecutter.vector_db -%}
from typing import Any, Callable, Optional, Type

from ...domain.vector import VectorDBConfig
from .base import BaseVectorDatabase


class VectorDBFactory:
    """Factory for creating vector database instances via a provider registry.

    Implements the Registry pattern so new providers can be added without
    modifying the factory logic.
    """

    _registry: dict[str, Type[Any]] = {}

    @classmethod
    def register(cls, provider: str) -> Callable:
        """Decorator to register a database class for a specific provider.

        Args:
            provider: The provider identifier (e.g., 'milvus').

        Returns:
            The decorator function.
        """

        def decorator(db_cls: Type[Any]) -> Type[Any]:
            cls._registry[provider] = db_cls
            return db_cls

        return decorator

    @classmethod
    def create(
        cls,
        provider: str,
        config: Optional[VectorDBConfig] = None,
        **kwargs: Any,
    ) -> BaseVectorDatabase:
        """Create an instance of the requested vector database adapter.

        Args:
            provider: The provider identifier (must be registered).
            config: The configuration object.
            **kwargs: Additional overrides passed to the adapter
                constructor.

        Returns:
            An instance of the specific vector database adapter.

        Raises:
            ValueError: If the provider is not registered.
        """
        if provider not in cls._registry:
            raise ValueError(
                f"Provider '{provider}' is not registered."
            )
        db_cls = cls._registry[provider]
        return db_cls(config=config, **kwargs)
{%- endif -%}
