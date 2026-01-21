from typing import Any, Callable, Dict, Optional, Type

from ...domain.llm.protocols import ModelConfig


class LlmFactory:
    """Factory for creating LLM instances based on a provider registry.

    This class implements the Registry pattern to allow dynamic registration
    of new LLM providers without modifying the factory logic.
    """

    _registry: Dict[str, Type[Any]] = {}

    @classmethod
    def register(cls, provider: str) -> Callable:
        """Decorator to register a model wrapper class for a specific provider.

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
    def create(cls, provider: str, config: Optional[ModelConfig] = None, **kwargs: Any) -> Any:
        """Create an instance of the requested LLM model.

        Args:
            provider: The provider identifier (must be registered).
            config: The configuration object.
            **kwargs: Additional overrides passed to the model constructor.

        Returns:
            An instance of the specific model wrapper.

        Raises:
            ValueError: If the provider is not registered.
        """
        if provider not in cls._registry:
            raise ValueError(f"Provider '{provider}' is not registered.")

        model_cls = cls._registry[provider]
        return model_cls(config=config, **kwargs)
