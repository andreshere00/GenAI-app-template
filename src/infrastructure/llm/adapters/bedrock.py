from typing import Any, Optional, Union

from langchain_aws import ChatBedrock
from pydantic import SecretStr

from ....domain.llm.constants import BEDROCK_PARAM_MAP
from ....domain.llm.protocols import ModelConfig
from ....domain.llm.types import LLMProvider
from ..base import BaseLlm
from ..factory import LlmFactory
from ...utils import resolve_parameters


@LlmFactory.register(LLMProvider.AWS)
class BedrockModel(BaseLlm):
    """Wrapper class for LangChain's ChatBedrock with flexible configuration.

    This class implements a configuration merging strategy, allowing instantiation
    via a structured configuration object, direct keyword arguments, or a mix of both.

    Attributes:
        client (ChatBedrock): The instantiated LangChain Bedrock client.
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        *,
        model: Optional[str] = None,
        api_key: Optional[Union[str, SecretStr]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the BedrockModel wrapper.

        The initialization logic prioritizes explicit keyword arguments over the
        configuration object.

        Args:
            config: An object adhering to the ModelConfig protocol.
            model: The model ID (e.g., "anthropic.claude-v2").
            api_key: API Key (Unused for Bedrock; relies on AWS Credentials/Profile).
            **kwargs: Additional arguments to pass to ChatBedrock.

        Raises:
            ValidationError: If required parameters are missing (handled by LangChain).
        """
        params: dict[str, Any] = self.resolve_parameters(
            config, model=model, api_key=api_key, **kwargs
        )

        for config_key, langchain_key in BEDROCK_PARAM_MAP.items():
            if config_key in params:
                params[langchain_key] = params.pop(config_key)

        params.pop("api_key", None)

        self.client = ChatBedrock(**params)