from typing import Any, Optional, Union

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from ....domain.llm.constants import OPENAI_PARAM_MAP
from ....domain.llm.protocols import ModelConfig
from ....domain.llm.types import LLMProvider
from ..base import BaseLlm
from ..factory import LlmFactory
from ...utils import resolve_parameters


@LlmFactory.register(LLMProvider.OPENAI)
class OpenAIModel(BaseLlm):
    """Wrapper class for LangChain's ChatOpenAI with flexible configuration.

    This class implements a configuration merging strategy, allowing instantiation
    via a structured configuration object, direct keyword arguments, or a mix of both.

    Attributes:
        client (ChatOpenAI): The instantiated LangChain OpenAI client.
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        *,
        model: Optional[str] = None,
        api_key: Optional[Union[str, SecretStr]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenAIModel wrapper.

        The initialization logic prioritizes explicit keyword arguments over the
        configuration object. If both are missing for a required field, LangChain's
        validation will raise the appropriate error.

        Args:
            config: An object adhering to the ModelConfig protocol.
            model: The name of the model (e.g., "gpt-4").
            api_key: The OpenAI API key.
            **kwargs: Additional arguments to pass to ChatOpenAI (e.g., temperature).

        Raises:
            ValidationError: If required parameters are missing (handled by LangChain).
        """
        params: dict[str, Any] = resolve_parameters(
            config, model=model, api_key=api_key, **kwargs
        )

        for config_key, langchain_key in OPENAI_PARAM_MAP.items():
            if config_key in params:
                params[langchain_key] = params.pop(config_key)

        self.client = ChatOpenAI(**params)
