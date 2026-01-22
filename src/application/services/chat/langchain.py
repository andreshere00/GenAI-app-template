from typing import Any, AsyncGenerator, List
from .base import BaseChatService
from ....domain.chat.types import ChatMessage


class LangChainChatService(BaseChatService):
    """LangChain implementation of the chat service."""

    async def _chat_direct(self, prompt_path: str, variables: dict[str, Any]) -> ChatMessage:
        """Executes a direct call using LangChain's ainvoke."""
        prompt_entity = self.repository.get_prompt(prompt_path, variables)

        response = await self.llm.client.ainvoke(prompt_entity.content)

        message = ChatMessage(role="assistant", content=response.content)
        self.history.append(message)
        return message

    async def _chat_stream(
        self, prompt_path: str, variables: dict[str, Any]
    ) -> AsyncGenerator[ChatMessage, None]:
        """Executes a streaming call using LangChain's astream."""
        prompt_entity = self.repository.get_prompt(prompt_path, variables)

        async for chunk in self.llm.client.astream(prompt_entity.content):
            yield ChatMessage(role="assistant", content=chunk.content)
