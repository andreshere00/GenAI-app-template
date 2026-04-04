from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import MagicMock
from uuid import UUID

from src.infrastructure.embedding.base import BaseEmbedding

# ---- Mocks, fixtures & helpers ---- #


@dataclass
class MockSplitterOutput:
    """Mock implementation of SplitterMR SplitterOutput for testing."""

    chunks: list[str] = field(default_factory=list)
    chunk_id: list[str] = field(default_factory=list)
    document_name: Optional[str] = None
    document_path: str = ""
    document_id: Optional[str] = None
    conversion_method: Optional[str] = None
    reader_method: Optional[str] = None
    ocr_method: Optional[str] = None
    split_method: str = ""
    split_params: Optional[dict[str, Any]] = None
    metadata: Optional[dict[str, Any]] = None


class ConcreteEmbedding(BaseEmbedding):
    """Concrete subclass of BaseEmbedding for testing."""

    def __init__(self, mock_client: Any) -> None:
        self.client = mock_client


# ---- Happy path ---- #


class TestBaseEmbedding:
    def test_embed_valid_output_returns_correct_vectors(self):
        """Test that embed delegates to embed_documents and returns vectors."""
        mock_client = MagicMock()
        mock_client.embed_documents.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]

        splitter_output = MockSplitterOutput(
            chunks=["hello world", "foo bar"],
            chunk_id=["c1", "c2"],
            document_path="/docs/test.pdf",
            split_method="character",
        )

        model = ConcreteEmbedding(mock_client)
        result = model.embed(splitter_output)

        mock_client.embed_documents.assert_called_once_with(
            ["hello world", "foo bar"]
        )
        assert result.embeddings == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    def test_embed_valid_output_carries_all_metadata(self):
        """Test that all splitter metadata is propagated to the result."""
        mock_client = MagicMock()
        mock_client.embed_documents.return_value = [[0.1]]

        splitter_output = MockSplitterOutput(
            chunks=["chunk"],
            chunk_id=["id-1"],
            document_name="report.pdf",
            document_path="/data/report.pdf",
            document_id="doc-abc",
            conversion_method="pdf",
            reader_method="docling",
            ocr_method="tesseract",
            split_method="sentence",
            split_params={"chunk_size": 200},
            metadata={"author": "test"},
        )

        result = ConcreteEmbedding(mock_client).embed(splitter_output)

        assert result.chunks == ["chunk"]
        assert result.chunk_id == ["id-1"]
        assert result.document_name == "report.pdf"
        assert result.document_path == "/data/report.pdf"
        assert result.document_id == "doc-abc"
        assert result.conversion_method == "pdf"
        assert result.reader_method == "docling"
        assert result.ocr_method == "tesseract"
        assert result.split_method == "sentence"
        assert result.split_params == {"chunk_size": 200}
        assert result.metadata == {"author": "test"}

    def test_embed_valid_output_generates_valid_uuid(self):
        """Test that embedding_id is a valid UUID4 string."""
        mock_client = MagicMock()
        mock_client.embed_documents.return_value = [[0.1]]

        splitter_output = MockSplitterOutput(
            chunks=["text"],
            chunk_id=["id"],
            document_path="/f.txt",
            split_method="word",
        )

        result = ConcreteEmbedding(mock_client).embed(splitter_output)

        UUID(result.embedding_id, version=4)

    # ---- Edge cases ---- #

    def test_embed_optional_fields_missing_returns_none(self):
        """Test that missing optional fields default to None."""
        mock_client = MagicMock()
        mock_client.embed_documents.return_value = [[0.5]]

        splitter_output = MockSplitterOutput(
            chunks=["only chunk"],
            chunk_id=["only-id"],
            document_path="/minimal.txt",
            split_method="char",
        )

        result = ConcreteEmbedding(mock_client).embed(splitter_output)

        assert result.document_name is None
        assert result.document_id is None
        assert result.conversion_method is None
        assert result.reader_method is None
        assert result.ocr_method is None
        assert result.split_params is None
        assert result.metadata is None

    def test_embed_multiple_calls_generate_unique_ids(self):
        """Test that consecutive embed calls produce different IDs."""
        mock_client = MagicMock()
        mock_client.embed_documents.return_value = [[0.1]]

        splitter_output = MockSplitterOutput(
            chunks=["a"],
            chunk_id=["x"],
            document_path="/a.txt",
            split_method="char",
        )

        model = ConcreteEmbedding(mock_client)
        r1 = model.embed(splitter_output)
        r2 = model.embed(splitter_output)

        assert r1.embedding_id != r2.embedding_id
