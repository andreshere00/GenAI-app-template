from __future__ import annotations

from typing import Any
from uuid import uuid4

from ...domain.embedding.types import Embedding as EmbeddingDTO


class BaseEmbedding:
    """Base class for all embedding model wrappers.

    Provides a common ``embed`` method that delegates vector generation
    to the provider-specific client and wraps the result into an
    ``Embedding`` dataclass.

    Attributes:
        client: The LangChain embeddings instance set by subclasses.
    """

    client: Any

    def embed(self, splitter_output: Any) -> EmbeddingDTO:
        """Produce embeddings from a SplitterMR ``SplitterOutput``.

        Calls the underlying LangChain client's ``embed_documents``
        method and combines the resulting vectors with the splitter
        metadata.

        Args:
            splitter_output: A SplitterMR ``SplitterOutput`` instance
                (or any object exposing the same attributes).

        Returns:
            An Embedding dataclass containing vectors and document
            metadata.
        """
        vectors: list[list[float]] = self.client.embed_documents(
            splitter_output.chunks
        )
        return EmbeddingDTO(
            embeddings=vectors,
            embedding_id=str(uuid4()),
            chunks=splitter_output.chunks,
            chunk_id=splitter_output.chunk_id,
            document_name=getattr(
                splitter_output, "document_name", None
            ),
            document_path=splitter_output.document_path,
            document_id=getattr(
                splitter_output, "document_id", None
            ),
            conversion_method=getattr(
                splitter_output, "conversion_method", None
            ),
            reader_method=getattr(
                splitter_output, "reader_method", None
            ),
            ocr_method=getattr(
                splitter_output, "ocr_method", None
            ),
            split_method=splitter_output.split_method,
            split_params=getattr(
                splitter_output, "split_params", None
            ),
            metadata=getattr(splitter_output, "metadata", None),
        )
