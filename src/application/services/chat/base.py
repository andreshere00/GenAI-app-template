from typing import Any, AsyncGenerator, List, Optional, Union
from ....domain.chat.types import ChatMessage, ChatMode
from ....domain.chat.protocols import ChatConfig
from ....infrastructure.llm.base import BaseLlm
from ....infrastructure.prompt.repositories.base import BasePromptRepository


class BaseChatService:
    """Base class for managing chat conversations with memory and flexible execution.

    This service orchestrates prompt retrieval, LLM interaction, and history management.
    """

    def __init__(
        self,
        llm: BaseLlm,
        repository: BasePromptRepository,
        config: Optional[ChatConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the chat service with dependencies and configuration.

        Args:
            llm: The LLM wrapper instance.
            repository: The prompt repository instance.
            config: A configuration object.
            **kwargs: Explicit configuration overrides.
        """
        self.llm = llm
        self.repository = repository
        self.history: List[ChatMessage] = []
        self.params = self._resolve_config(config, **kwargs)

    def _resolve_config(self, config: Optional[ChatConfig], **overrides: Any) -> dict[str, Any]:
        """Merges configuration object with explicit parameters."""
        base = {"mode": ChatMode.DIRECT, "max_history": 10}
        if config:
            base.update({k: v for k, v in config.__dict__.items() if v is not None})
        base.update(overrides)
        return base

    async def chat(
        self, prompt_path: str, variables: dict[str, Any]
    ) -> Union[ChatMessage, AsyncGenerator[ChatMessage, None], List[ChatMessage]]:
        """
        Main entry point to execute the chat based on the configured mode.

        Args:
            prompt_path: The path to the prompt template.
            variables: Variables to inject into the prompt template.

        Returns:
            Depending on the mode, returns a ChatMessage, an async generator of ChatMessages,
            or a list of ChatMessages.
        """
        mode = self.params.get("mode")

        if mode == ChatMode.STREAM:
            return self._chat_stream(prompt_path, variables)
        elif mode == ChatMode.BATCH:
            return await self._chat_batch(prompt_path, variables)
        return await self._chat_direct(prompt_path, variables)

    async def _chat_direct(self, prompt_path: str, variables: dict[str, Any]) -> ChatMessage:
        """Internal method for direct response."""
        raise NotImplementedError

    async def _chat_stream(
        self, prompt_path: str, variables: dict[str, Any]
    ) -> AsyncGenerator[ChatMessage, None]:
        """Internal method for streaming response."""
        raise NotImplementedError
