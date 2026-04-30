from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional

from ....domain.vector import (
    CollectionConfigDTO,
    DistanceMetric,
    VectorRecordDTO,
    VectorSearchResultDTO,
)
from ....infrastructure.embedding.base import BaseEmbedding
from ....infrastructure.vector.base import BaseVectorDatabase
from ..chat.base import BaseChatService


@dataclass
class RagIngestionResult:
    """Result summary after ingesting a document into the vector database."""

    document_id: str
    collection_name: str
    chunks_count: int
    record_ids: list[str]


@dataclass
class RagAnswer:
    """Grounded answer returned by the RAG service."""

    answer: Any
    matches: list[VectorSearchResultDTO]
    context: str


class BaseRagService:
    """Base application service orchestrating a RAG pipeline."""

    def __init__(
        self,
        vector_db: BaseVectorDatabase,
        embedding_model: BaseEmbedding,
        chat_service: BaseChatService,
        config: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize dependencies and RAG configuration.

        Args:
            vector_db: Vector database adapter used for persistence/retrieval.
            embedding_model: Embedding adapter used to encode text chunks.
            chat_service: Chat service used to generate grounded answers.
            config: Optional configuration dictionary.
            **kwargs: Explicit configuration overrides.
        """
        self.vector_db = vector_db
        self.embedding_model = embedding_model
        self.chat_service = chat_service
        self.params = self._resolve_config(config, **kwargs)

    def _resolve_config(
        self,
        config: Optional[dict[str, Any]],
        **overrides: Any,
    ) -> dict[str, Any]:
        """Merge configuration with explicit overrides."""
        base = {
            "collection_name": "documents",
            "top_k": 4,
            "prompt_path": "prompts/rag/default.md",
            "reader_method": "markitdown",
            "splitter_method": "recursive",
            "chunk_size": 1000,
            "chunk_overlap": 100,
            "distance_metric": DistanceMetric.COSINE,
        }
        if config:
            base.update({k: v for k, v in config.items() if v is not None})
        base.update(overrides)
        return base

    def _create_reader(self) -> Any:
        """Create the configured SplitterMR reader instance."""
        reader_method = str(self.params.get("reader_method", "markitdown")).lower()
        if reader_method != "markitdown":
            raise ValueError(f"Unsupported reader method: {reader_method}")

        from splitter_mr.reader import MarkItDownReader

        return MarkItDownReader()

    def _create_splitter(self) -> Any:
        """Create the configured SplitterMR splitter instance."""
        splitter_method = str(self.params.get("splitter_method", "recursive")).lower()
        if splitter_method != "recursive":
            raise ValueError(f"Unsupported splitter method: {splitter_method}")

        from splitter_mr.splitter import RecursiveCharacterSplitter

        return RecursiveCharacterSplitter(
            chunk_size=int(self.params.get("chunk_size", 1000)),
            chunk_overlap=int(self.params.get("chunk_overlap", 100)),
        )

    def _build_context(self, matches: list[VectorSearchResultDTO]) -> str:
        """Build a text context from retrieved records."""
        context_chunks = [
            str(match.payload.get("chunk", "")).strip()
            for match in matches
            if match.payload.get("chunk")
        ]
        return "\n\n".join(context_chunks)

    def _embed_query(self, query: str) -> list[float]:
        """Embed a query string using the configured embedding adapter."""
        if hasattr(self.embedding_model.client, "embed_query"):
            return self.embedding_model.client.embed_query(query)

        splitter_output = SimpleNamespace(
            chunks=[query],
            chunk_id=["query-0"],
            document_name="query",
            document_path="",
            document_id="query",
            conversion_method=None,
            reader_method=None,
            ocr_method=None,
            split_method="query",
            split_params={},
            metadata={},
        )
        output = self.embedding_model.embed(splitter_output)
        return output.embeddings[0] if output.embeddings else []

    def ingest_document(
        self,
        document_path: str,
        collection_name: Optional[str] = None,
        ensure_collection: bool = True,
        **kwargs: Any,
    ) -> RagIngestionResult:
        """Read, chunk, embed and persist one document in the vector DB.

        Args:
            document_path: Path or URL passed to the configured reader.
            collection_name: Optional target collection/index override.
            ensure_collection: Create the collection if missing.
            **kwargs: Provider-specific upsert options.

        Returns:
            Summary with document id, collection and inserted record IDs.
        """
        reader = self._create_reader()
        splitter = self._create_splitter()
        reader_output = reader.read(document_path)
        splitter_output = splitter.split(reader_output)
        embedding_output = self.embedding_model.embed(splitter_output)

        target_collection = collection_name or str(self.params["collection_name"])
        vectors = embedding_output.embeddings
        record_ids = list(splitter_output.chunk_id)
        dimension = len(vectors[0]) if vectors else None

        if ensure_collection and not self.vector_db.has_collection(target_collection):
            distance_metric = self.params["distance_metric"]
            metric_value = (
                distance_metric.value
                if isinstance(distance_metric, DistanceMetric)
                else str(distance_metric)
            )
            self.vector_db.create_collection(
                CollectionConfigDTO(
                    name=target_collection,
                    dimension=dimension,
                    metric=metric_value,
                )
            )

        records = [
            VectorRecordDTO(
                id=record_id,
                vector=vector,
                payload={
                    "chunk": chunk,
                    "chunk_id": record_id,
                    "document_name": splitter_output.document_name,
                    "document_path": splitter_output.document_path,
                    "document_id": splitter_output.document_id,
                    "conversion_method": splitter_output.conversion_method,
                    "reader_method": splitter_output.reader_method,
                    "ocr_method": splitter_output.ocr_method,
                    "split_method": splitter_output.split_method,
                    "split_params": splitter_output.split_params or {},
                    "metadata": splitter_output.metadata or {},
                },
            )
            for record_id, chunk, vector in zip(
                record_ids,
                splitter_output.chunks,
                vectors,
            )
        ]
        self.vector_db.upsert(target_collection, records, **kwargs)

        return RagIngestionResult(
            document_id=str(splitter_output.document_id),
            collection_name=target_collection,
            chunks_count=len(records),
            record_ids=record_ids,
        )

    async def ask(
        self,
        question: str,
        prompt_path: Optional[str] = None,
        collection_name: Optional[str] = None,
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> RagAnswer:
        """Retrieve relevant chunks and generate a grounded chat response.

        Args:
            question: User query to answer.
            prompt_path: Prompt path for chat service.
            collection_name: Optional collection/index override.
            top_k: Optional retrieval limit override.
            **kwargs: Additional search options for vector DB.

        Returns:
            Grounded answer, retrieval matches and built context string.
        """
        target_collection = collection_name or str(self.params["collection_name"])
        limit = int(top_k or self.params["top_k"])
        query_vector = self._embed_query(question)
        matches = self.vector_db.search(
            collection_name=target_collection,
            query_vector=query_vector,
            limit=limit,
            **kwargs,
        )
        context = self._build_context(matches)

        answer = await self.chat_service.chat(
            prompt_path or str(self.params["prompt_path"]),
            {"question": question, "context": context},
        )
        return RagAnswer(answer=answer, matches=matches, context=context)
