{%- if "chat" in cookiecutter.services -%}
import asyncio
from typing import AsyncGenerator
from unittest.mock import MagicMock

from src.infrastructure.llm.adapters.openai import OpenAIModel
from src.infrastructure.prompt.repositories.base import BasePromptRepository
from src.infrastructure.prompt.storage.adapters.local import LocalStorageAdapter
from src.application.services.chat.langchain import LangChainChatService
from src.domain.chat.types import ChatMode, ChatMessage
from src.domain.prompt.types import PromptTemplate


async def run_mocked_support_chat_use_case() -> None:
    """E2E simulation using mocks for storage and LLM streaming.

    This use case demonstrates how to test the chat service integration
    without making real requests to external providers.
    """

    # 1. Mock Storage (Simulates the .txt file on disk)
    mock_storage = MagicMock(spec=LocalStorageAdapter)
    mock_storage.load_template.return_value = PromptTemplate(
        content="Hello {{user_name}}, welcome to {{company}}. Query: {{user_query}}",
        path="support_agent.txt",
    )

    repository = BasePromptRepository(storage_adapter=mock_storage)

    # 2. Mock LLM (Simulates the OpenAI/LangChain response)
    mock_llm = MagicMock(spec=OpenAIModel)
    mock_client = MagicMock()

    # Simulate the asynchronous generator that LangChain would use in stream mode
    async def mock_astream_generator(*args, **kwargs):
        chunks = ["Sure! ", "To reset ", "your password, ", "click ", "on the link."]
        for text in chunks:
            # LangChain returns objects containing a .content attribute
            chunk_mock = MagicMock()
            chunk_mock.content = text
            yield chunk_mock

    # Assign the simulated generator to the client's astream method
    mock_client.astream.side_effect = mock_astream_generator
    mock_llm.client = mock_client

    # 3. Initialize Chat Service with test configuration
    chat_service = LangChainChatService(
        llm=mock_llm, repository=repository, mode=ChatMode.STREAM, max_history=5
    )

    # 4. Input variables for the prompt
    input_vars = {
        "company": "TechFlow Inc.",
        "user_name": "andreshere00",
        "user_query": "How can I reset my password?",
    }

    # 5. Execution and console output
    print("Assistant: ", end="", flush=True)

    # The chat method returns an AsyncGenerator due to STREAM mode
    response_stream: AsyncGenerator[ChatMessage, None] = await chat_service.chat(
        prompt_path="support_agent.txt", variables=input_vars
    )

    async for chunk in response_stream:
        print(chunk.content, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    asyncio.run(run_mocked_support_chat_use_case())
{% endif %}