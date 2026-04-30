from typing import Any, Optional, Union

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from ....domain.llm.constants import GEMINI_PARAM_MAP
from ....domain.llm.protocols import ModelConfig
from ....domain.llm.types import LLMProvider
from ..base import BaseLlm
from ..factory import LlmFactory
from ...utils import resolve_parameters


@LlmFactory.register(LLMProvider.GOOGLE)
class GeminiModel(BaseLlm):
    """Wrapper class for LangChain's ChatGoogleGenerativeAI with flexible configuration.

    This class implements a configuration merging strategy, allowing instantiation
    via a structured configuration object, direct keyword arguments, or a mix of both.

    Attributes:
        client (ChatGoogleGenerativeAI): The instantiated LangChain Gemini client.
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        *,
        model: Optional[str] = None,
        api_key: Optional[Union[str, SecretStr]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the GeminiModel wrapper.

        The initialization logic prioritizes explicit keyword arguments over the
        configuration object.

        Args:
            config: An object adhering to the ModelConfig protocol.
            model: The name of the model (e.g., "gemini-pro").
            api_key: The Google API key.
            **kwargs: Additional arguments to pass to ChatGoogleGenerativeAI.

        Raises:
            ValidationError: If required parameters are missing (handled by LangChain).
        """
        params: dict[str, Any] = resolve_parameters(
            config, model=model, api_key=api_key, **kwargs
        )

        for config_key, langchain_key in GEMINI_PARAM_MAP.items():
            if config_key in params:
                params[langchain_key] = params.pop(config_key)

        self.client = ChatGoogleGenerativeAI(**params)
