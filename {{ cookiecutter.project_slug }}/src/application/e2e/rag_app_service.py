{%- if "rag" in cookiecutter.services -%}
import asyncio

from src.application.services.chat.langchain import LangChainChatService
from src.application.services.rag import BaseRagService
from src.domain.chat.types import ChatMode
from src.infrastructure.embedding.adapters.openai import OpenAIEmbeddingModel
from src.infrastructure.llm.adapters.openai import OpenAIModel
from src.infrastructure.prompt.repositories.base import BasePromptRepository
from src.infrastructure.prompt.storage.adapters.local import LocalStorageAdapter
from src.infrastructure.vector.adapters.qdrant_db import QdrantVectorDatabase


async def run_rag_use_case() -> None:
    """Example E2E flow that ingests a document and answers a question."""
    storage = LocalStorageAdapter(base_path="prompts/templates")
    repository = BasePromptRepository(storage_adapter=storage)
    llm = OpenAIModel(model="gpt-4", temperature=0.2)
    chat_service = LangChainChatService(
        llm=llm,
        repository=repository,
        mode=ChatMode.DIRECT,
        max_history=5,
    )

    vector_db = QdrantVectorDatabase(host="localhost", port=6333)
    embedding_model = OpenAIEmbeddingModel(model="text-embedding-3-small")
    rag_service = BaseRagService(
        vector_db=vector_db,
        embedding_model=embedding_model,
        chat_service=chat_service,
        collection_name="support_docs",
        top_k=4,
        prompt_path="rag/default.txt",
    )

    rag_service.ingest_document(document_path="docs/support_policy.pdf")
    answer = await rag_service.ask("What is the password reset process?")
    print(answer.answer)


if __name__ == "__main__":
    asyncio.run(run_rag_use_case())
{%- endif -%}
