import pytest
from unittest.mock import MagicMock, AsyncMock

from src.application.services.chat.base import BaseChatService
from src.domain.chat.types import ChatMessage, ChatMode
from src.domain.chat.protocols import ChatConfig


class MockChatService(BaseChatService):
    """Minimal implementation of BaseChatService for testing purposes."""

    async def _chat_direct(self, prompt_path: str, variables: dict) -> ChatMessage:
        return ChatMessage(role="assistant", content="direct response")

    async def _chat_stream(self, prompt_path: str, variables: dict):
        yield ChatMessage(role="assistant", content="stream response")

    async def _chat_batch(self, prompt_path: str, variables: dict):
        return [ChatMessage(role="assistant", content="batch response")]


@pytest.fixture
def mock_dependencies() -> tuple[MagicMock, MagicMock]:
    """Provides mocked LLM and Repository dependencies."""
    return MagicMock(), MagicMock()


def test_resolve_config_no_config_returns_defaults(
    mock_dependencies: tuple[MagicMock, MagicMock],
) -> None:
    """Tests if defaults are applied when no configuration is provided."""
    llm, repo = mock_dependencies
    service = BaseChatService(llm=llm, repository=repo)

    assert service.params["mode"] == ChatMode.DIRECT
    assert service.params["max_history"] == 10


def test_resolve_config_with_overrides_precedence_returns_merged_values(
    mock_dependencies: tuple[MagicMock, MagicMock],
) -> None:
    """Tests if explicit kwargs take precedence over default values."""
    llm, repo = mock_dependencies
    service = BaseChatService(
        llm=llm, repository=repo, mode=ChatMode.STREAM, max_history=20
    )

    assert service.params["mode"] == ChatMode.STREAM
    assert service.params["max_history"] == 20


@pytest.mark.asyncio
async def test_chat_direct_mode_calls_internal_direct_method(
    mock_dependencies: tuple[MagicMock, MagicMock],
) -> None:
    """Verifies that DIRECT mode routes to the correct internal handler."""
    llm, repo = mock_dependencies
    service = MockChatService(llm=llm, repository=repo, mode=ChatMode.DIRECT)

    result = await service.chat("path", {})

    assert isinstance(result, ChatMessage)
    assert result.content == "direct response"


@pytest.mark.asyncio
async def test_chat_stream_mode_returns_async_generator(
    mock_dependencies: tuple[MagicMock, MagicMock],
) -> None:
    """Verifies that STREAM mode returns a generator for iterative processing."""
    llm, repo = mock_dependencies
    service = MockChatService(llm=llm, repository=repo, mode=ChatMode.STREAM)

    generator = await service.chat("path", {})

    chunks = []
    async for chunk in generator:
        chunks.append(chunk)

    assert len(chunks) == 1
    assert chunks[0].content == "stream response"


@pytest.mark.asyncio
async def test_chat_batch_mode_returns_list_of_messages(
    mock_dependencies: tuple[MagicMock, MagicMock],
) -> None:
    """Verifies that BATCH mode routes to the batch handler returning a list."""
    llm, repo = mock_dependencies
    service = MockChatService(llm=llm, repository=repo, mode=ChatMode.BATCH)

    result = await service.chat("path", {})

    assert isinstance(result, list)
    assert result[0].content == "batch response"
