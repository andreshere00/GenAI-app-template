from typing import Any, Optional, Union

from langchain_ollama import ChatOllama
from pydantic import SecretStr

from ....domain.llm.protocols import ModelConfig
from ....domain.llm.types import LLMProvider
from ..base import BaseLlm
from ..factory import LlmFactory
from ..utils import resolve_parameters


@LlmFactory.register(LLMProvider.HUGGINGFACE)
class OllamaModel(BaseLlm):
    """Wrapper class for LangChain's ChatOllama with flexible configuration.

    This class implements a configuration merging strategy, allowing instantiation
    via a structured configuration object, direct keyword arguments, or a mix of both.

    Attributes:
        client (ChatOllama): The instantiated LangChain Ollama client.
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        *,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[Union[str, SecretStr]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the OllamaModel wrapper.

        The initialization logic prioritizes explicit keyword arguments over the
        configuration object.

        Args:
            config: An object adhering to the ModelConfig protocol.
            model: The name of the model (e.g., "llama3").
            base_url: The URL of the Ollama server (e.g., "http://localhost:11434").
            api_key: Unused for Ollama, but accepted for protocol compatibility.
            **kwargs: Additional arguments to pass to ChatOllama (e.g., temperature).

        Raises:
            ValidationError: If required parameters are missing (handled by LangChain).
        """
        params: dict[str, Any] = self._resolve_parameters(
            config, model=model, base_url=base_url, api_key=api_key, **kwargs
        )

        # Cleanup: Remove 'api_key' if present, as ChatOllama does not expect it.
        params.pop("api_key", None)

        self.client = ChatOllama(**params)

    def _resolve_parameters(
        self, config: Optional[ModelConfig], **overrides: Any
    ) -> dict[str, Any]:
        """Merge configuration object attributes with explicit overrides.

        Args:
            config: The source configuration object.
            **overrides: Dictionary of arguments passed directly to __init__.

        Returns:
            A dictionary containing the final non-None parameters for instantiation.
        """
        return resolve_parameters(config, **overrides)
