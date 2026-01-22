import asyncio
from typing import AsyncGenerator

from src.infrastructure.llm.adapters.openai import OpenAIModel
from src.infrastructure.prompt.repositories.base import BasePromptRepository
from src.infrastructure.prompt.storage.adapters.local import LocalStorageAdapter
from src.application.services.chat.langchain import LangChainChatService
from src.domain.chat.types import ChatMode, ChatMessage


async def run_support_chat_use_case():
    """Example of an E2E chat interaction using the LangChainChatService."""

    # 1. Setup Infrastructure
    storage = LocalStorageAdapter(base_path="prompts/templates")
    repository = BasePromptRepository(storage_adapter=storage)

    # 2. Setup LLM (Using your OpenAIModel adapter)
    llm = OpenAIModel(model="gpt-4", temperature=0.7)

    # 3. Initialize Chat Service with specific configuration
    chat_service = LangChainChatService(
        llm=llm,
        repository=repository,
        mode=ChatMode.STREAM,  # We choose STREAM mode for a better UX
        max_history=5,  # Customize max history as needed
    )

    # 4. Define variables for the prompt
    input_vars: dict[str, str] = {
        "company": "TechFlow Inc.",
        "user_name": "andreshere00",
        "user_query": "How can I reset my password?",
    }

    # 5. Execute the chat (Asynchronous Stream)
    print(f"Assistant: ", end="", flush=True)

    # The chat method returns an AsyncGenerator because mode=STREAM
    response_stream: AsyncGenerator[ChatMessage, None] = await chat_service.chat(
        prompt_path="support_agent.txt", variables=input_vars
    )

    async for chunk in response_stream:
        print(chunk.content, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    asyncio.run(run_support_chat_use_case())
