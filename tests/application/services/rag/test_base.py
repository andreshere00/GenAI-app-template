from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from src.application.services.rag.base import BaseRagService
from src.domain.chat.types import ChatMessage
from src.domain.vector import CollectionConfigDTO, VectorRecordDTO, VectorSearchResultDTO


# ---- Mocks, fixtures & helpers ---- #
@dataclass
class DummySplitterOutput:
    chunks: list[str]
    chunk_id: list[str]
    document_name: str = "doc.md"
    document_path: str = "/tmp/doc.md"
    document_id: str = "doc-1"
    conversion_method: str | None = "markitdown"
    reader_method: str | None = "markitdown"
    ocr_method: str | None = None
    split_method: str = "recursive"
    split_params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class DummyVectorDB:
    def __init__(self) -> None:
        self.collections: set[str] = set()
        self.upsert_calls: list[dict[str, Any]] = []
        self.search_results: list[VectorSearchResultDTO] = []

    def has_collection(self, name: str) -> bool:
        return name in self.collections

    def create_collection(self, config: CollectionConfigDTO) -> None:
        self.collections.add(config.name)

    def upsert(self, collection_name: str, records: list[VectorRecordDTO], **kwargs: Any) -> None:
        self.upsert_calls.append({"collection_name": collection_name, "records": records})

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 5,
        **kwargs: Any,
    ) -> list[VectorSearchResultDTO]:
        return self.search_results[:limit]


class DummyEmbeddingModel:
    def __init__(self) -> None:
        self.client = SimpleNamespace(embed_query=lambda question: [0.1, 0.2, 0.3])

    def embed(self, splitter_output: DummySplitterOutput) -> Any:
        return SimpleNamespace(embeddings=[[0.1, 0.2, 0.3] for _ in splitter_output.chunks])


class DummyChatService:
    async def chat(self, prompt_path: str, variables: dict[str, Any]) -> ChatMessage:
        return ChatMessage(role="assistant", content=f"{prompt_path}:{variables['question']}")


@pytest.fixture
def rag_service(monkeypatch: pytest.MonkeyPatch) -> BaseRagService:
    vector_db = DummyVectorDB()
    embedding = DummyEmbeddingModel()
    chat_service = DummyChatService()
    service = BaseRagService(
        vector_db=vector_db,
        embedding_model=embedding,
        chat_service=chat_service,
        collection_name="docs",
        top_k=2,
        prompt_path="rag/default.txt",
    )
    splitter_output = DummySplitterOutput(
        chunks=["alpha", "beta"],
        chunk_id=["c1", "c2"],
    )
    monkeypatch.setattr(service, "_create_reader", lambda: SimpleNamespace(read=lambda _: "reader-output"))
    monkeypatch.setattr(
        service,
        "_create_splitter",
        lambda: SimpleNamespace(split=lambda _: splitter_output),
    )
    return service


# ---- Happy path ---- #
def test_ingest_document_valid_document_upserts_all_chunks(rag_service: BaseRagService) -> None:
    result = rag_service.ingest_document(document_path="docs/file.pdf")
    vector_db = rag_service.vector_db

    assert result.collection_name == "docs"
    assert result.chunks_count == 2
    assert len(vector_db.upsert_calls) == 1
    assert len(vector_db.upsert_calls[0]["records"]) == 2


@pytest.mark.asyncio
async def test_ask_with_matches_returns_grounded_answer(rag_service: BaseRagService) -> None:
    vector_db = rag_service.vector_db
    vector_db.search_results = [
        VectorSearchResultDTO(id="c1", score=0.9, payload={"chunk": "policy chunk"}),
        VectorSearchResultDTO(id="c2", score=0.8, payload={"chunk": "support chunk"}),
    ]

    answer = await rag_service.ask(question="How do I reset my password?")

    assert answer.answer.content.endswith("How do I reset my password?")
    assert len(answer.matches) == 2
    assert "policy chunk" in answer.context


# ---- Error paths ---- #
def test_create_reader_invalid_method_raises_value_error() -> None:
    service = BaseRagService(
        vector_db=DummyVectorDB(),
        embedding_model=DummyEmbeddingModel(),
        chat_service=DummyChatService(),
        reader_method="invalid",
    )

    with pytest.raises(ValueError):
        service._create_reader()


# ---- Edge cases ---- #
@pytest.mark.asyncio
async def test_ask_without_matches_returns_empty_context(rag_service: BaseRagService) -> None:
    answer = await rag_service.ask(question="Unknown")

    assert answer.matches == []
    assert answer.context == ""
