from typing import Any, Optional, Union

from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr

from ....domain.llm.constants import AZURE_OPENAI_PARAM_MAP
from ....domain.llm.protocols import ModelConfig
from ....domain.llm.types import LLMProvider
from ..base import BaseLlm
from ..factory import LlmFactory


@LlmFactory.register(LLMProvider.AZURE)
class AzureOpenAIModel(BaseLlm):
    """Wrapper class for LangChain's AzureChatOpenAI with flexible configuration.

    This class adapts a generic ModelConfig into the specific parameters required
    by Azure, handling field mapping (e.g., base_url -> azure_endpoint).
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        *,
        model: Optional[str] = None,
        api_key: Optional[Union[str, SecretStr]] = None,
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the AzureOpenAIModel wrapper.

        Args:
            config: An object adhering to the ModelConfig protocol.
            model: The deployment name (maps to azure_deployment).
            api_key: The Azure API key.
            azure_endpoint: The Azure endpoint URL.
            api_version: The Azure API version (e.g., '2023-05-15').
            **kwargs: Additional arguments (temperature, etc.).
        """
        params = self._resolve_parameters(
            config,
            model=model,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            **kwargs,
        )

        for config_key, langchain_key in AZURE_OPENAI_PARAM_MAP.items():
            if config_key in params:
                if langchain_key not in params:
                    params[langchain_key] = params.pop(config_key)

        self.client = AzureChatOpenAI(**params)

    def _resolve_parameters(
        self, config: Optional[ModelConfig], **overrides: Any
    ) -> dict[str, Any]:
        """Merges configuration object attributes with explicit overrides.

        Args:
            config: The source configuration object.
            **overrides: Dictionary of arguments passed directly to __init__.

        Returns:
            A dictionary containing the final non-None parameters.
        """
        final_params: dict[str, Any] = {}

        if config:
            protocol_fields = ModelConfig.__annotations__.keys()
            for field in protocol_fields:
                if hasattr(config, field):
                    value = getattr(config, field)
                    if value is not None:
                        final_params[field] = value

        for key, value in overrides.items():
            if value is not None:
                final_params[key] = value

        return final_params
