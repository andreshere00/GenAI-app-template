from typing import Any, Optional, Union

from langchain_anthropic import ChatAnthropic
from pydantic import SecretStr

from ....domain.llm.constants import CLAUDE_PARAM_MAP
from ....domain.llm.protocols import ModelConfig
from ....domain.llm.types import LLMProvider
from ..base import BaseLlm
from ..factory import LlmFactory
from ..utils import resolve_parameters


@LlmFactory.register(LLMProvider.ANTHROPIC)
class AnthropicModel(BaseLlm):
    """Wrapper class for LangChain's ChatAnthropic (Claude) with flexible configuration.

    This class implements a configuration merging strategy, allowing instantiation
    via a structured configuration object, direct keyword arguments, or a mix of both.

    Attributes:
        client (ChatAnthropic): The instantiated LangChain Anthropic client.
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        *,
        model: Optional[str] = None,
        api_key: Optional[Union[str, SecretStr]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the AnthropicModel wrapper.

        The initialization logic prioritizes explicit keyword arguments over the
        configuration object.

        Args:
            config: An object adhering to the ModelConfig protocol.
            model: The name of the model (e.g., "claude-3-sonnet-20240229").
            api_key: The Anthropic API key.
            **kwargs: Additional arguments to pass to ChatAnthropic.

        Raises:
            ValidationError: If required parameters are missing (handled by LangChain).
        """
        params: dict[str, Any] = self._resolve_parameters(
            config, model=model, api_key=api_key, **kwargs
        )

        for config_key, langchain_key in CLAUDE_PARAM_MAP.items():
            if config_key in params:
                params[langchain_key] = params.pop(config_key)

        self.client = ChatAnthropic(**params)

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
