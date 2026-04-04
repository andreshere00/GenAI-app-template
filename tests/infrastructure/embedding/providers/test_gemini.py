from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from src.infrastructure.embedding.adapters.gemini import (
    GeminiEmbeddingModel,
)

# ---- Mocks, fixtures & helpers ---- #


@pytest.fixture
def mock_gemini_embeddings():
    """Patches GoogleGenerativeAIEmbeddings to prevent real API calls."""
    with patch(
        "src.infrastructure.embedding.adapters"
        ".gemini.GoogleGenerativeAIEmbeddings"
    ) as mock:
        yield mock


# ---- Happy path ---- #


class TestGeminiEmbeddingModel:
    def test_init_valid_config_sets_client_params(
        self, config_factory, mock_gemini_embeddings
    ):
        """Test instantiation using only the configuration object."""
        config = config_factory(
            api_key="sk-gemini", model="models/embedding-001"
        )

        GeminiEmbeddingModel(config=config)

        mock_gemini_embeddings.assert_called_once()
        kw = mock_gemini_embeddings.call_args.kwargs
        assert kw["google_api_key"] == "sk-gemini"
        assert kw["model"] == "models/embedding-001"

    def test_init_valid_kwargs_sets_client_params(
        self, mock_gemini_embeddings
    ):
        """Test instantiation using only explicit keyword arguments."""
        GeminiEmbeddingModel(
            api_key="sk-kwarg", model="models/text-embedding-004"
        )

        kw = mock_gemini_embeddings.call_args.kwargs
        assert kw["google_api_key"] == "sk-kwarg"
        assert kw["model"] == "models/text-embedding-004"

    def test_init_config_and_kwargs_prioritizes_kwargs(
        self, config_factory, mock_gemini_embeddings
    ):
        """Test that explicit kwargs override configuration values."""
        config = config_factory(
            api_key="sk-cfg", model="models/embedding-001"
        )

        GeminiEmbeddingModel(config=config, model="models/embedding-002")

        kw = mock_gemini_embeddings.call_args.kwargs
        assert kw["google_api_key"] == "sk-cfg"
        assert kw["model"] == "models/embedding-002"

    def test_init_remappable_keys_maps_to_langchain_params(
        self, mock_gemini_embeddings
    ):
        """Test that api_key is mapped to google_api_key."""
        GeminiEmbeddingModel(api_key="sk-123")

        kw = mock_gemini_embeddings.call_args.kwargs
        assert "api_key" not in kw
        assert kw["google_api_key"] == "sk-123"

    def test_init_secret_str_api_key_passes_raw(
        self, mock_gemini_embeddings
    ):
        """Test that Pydantic SecretStr is handled correctly."""
        secret = SecretStr("sk-secret")
        GeminiEmbeddingModel(api_key=secret)

        kw = mock_gemini_embeddings.call_args.kwargs
        assert kw["google_api_key"] == secret

    def test_register_class_loading_registers_in_factory(self):
        """Test that the class is registered in the factory."""
        from src.domain.embedding.types import EmbeddingProvider
        from src.infrastructure.embedding.factory import EmbeddingFactory

        assert EmbeddingProvider.GOOGLE in EmbeddingFactory._registry
        assert (
            EmbeddingFactory._registry[EmbeddingProvider.GOOGLE]
            == GeminiEmbeddingModel
        )

    def test_embed_delegates_to_client(self, mock_gemini_embeddings):
        """Test that embed() calls embed_documents and returns DTO."""
        mock_inst = mock_gemini_embeddings.return_value
        mock_inst.embed_documents.return_value = [[0.1, 0.2]]

        model = GeminiEmbeddingModel(api_key="sk-test")
        splitter = MagicMock(
            chunks=["hello"],
            chunk_id=["c1"],
            document_path="/t.txt",
            split_method="char",
        )
        result = model.embed(splitter)

        mock_inst.embed_documents.assert_called_once_with(["hello"])
        assert result.embeddings == [[0.1, 0.2]]
