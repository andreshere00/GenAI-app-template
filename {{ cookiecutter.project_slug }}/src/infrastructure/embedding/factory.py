from __future__ import annotations

from typing import Any, Callable, Optional, Type

from ...domain.embedding.protocols import EmbeddingConfig


class EmbeddingFactory:
    """Factory for creating embedding instances based on a provider registry.

    Implements the Registry pattern to allow dynamic registration of
    new embedding providers without modifying the factory logic.
    """

    _registry: dict[str, Type[Any]] = {}

    @classmethod
    def register(cls, provider: str) -> Callable:
        """Decorator to register an embedding wrapper for a provider.

        Args:
            provider: The provider identifier (e.g., 'openai').

        Returns:
            The decorator function.
        """

        def decorator(model_cls: Type[Any]) -> Type[Any]:
            cls._registry[provider] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(
        cls,
        provider: str,
        config: Optional[EmbeddingConfig] = None,
        **kwargs: Any,
    ) -> Any:
        """Create an instance of the requested embedding model.

        Args:
            provider: The provider identifier (must be registered).
            config: The configuration object.
            **kwargs: Additional overrides passed to the constructor.

        Returns:
            An instance of the specific embedding wrapper.

        Raises:
            ValueError: If the provider is not registered.
        """
        if provider not in cls._registry:
            raise ValueError(
                f"Embedding provider '{provider}' is not registered."
            )
        model_cls = cls._registry[provider]
        return model_cls(config=config, **kwargs)
