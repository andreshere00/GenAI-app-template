from unittest.mock import MagicMock, patch

import pytest

from src.infrastructure.embedding.adapters.ollama import (
    OllamaEmbeddingModel,
)

# ---- Mocks, fixtures & helpers ---- #


@pytest.fixture
def mock_ollama_embeddings():
    """Patches OllamaEmbeddings to prevent real API calls."""
    with patch(
        "src.infrastructure.embedding.adapters"
        ".ollama.OllamaEmbeddings"
    ) as mock:
        yield mock


# ---- Happy path ---- #


class TestOllamaEmbeddingModel:
    def test_init_valid_config_sets_client_params(
        self, config_factory, mock_ollama_embeddings
    ):
        """Test instantiation using only the configuration object."""
        config = config_factory(
            model="nomic-embed-text",
            base_url="http://localhost:11434",
            api_key="ignored-key",
        )

        OllamaEmbeddingModel(config=config)

        mock_ollama_embeddings.assert_called_once()
        kw = mock_ollama_embeddings.call_args.kwargs
        assert kw["model"] == "nomic-embed-text"
        assert kw["base_url"] == "http://localhost:11434"
        assert "api_key" not in kw

    def test_init_valid_kwargs_sets_client_params(
        self, mock_ollama_embeddings
    ):
        """Test instantiation using only explicit keyword arguments."""
        OllamaEmbeddingModel(
            model="mxbai-embed-large",
            base_url="http://ollama:11434",
        )

        kw = mock_ollama_embeddings.call_args.kwargs
        assert kw["model"] == "mxbai-embed-large"
        assert kw["base_url"] == "http://ollama:11434"

    def test_init_config_and_kwargs_prioritizes_kwargs(
        self, config_factory, mock_ollama_embeddings
    ):
        """Test that explicit kwargs override configuration values."""
        config = config_factory(
            model="nomic-embed-text",
            base_url="http://localhost:11434",
        )

        OllamaEmbeddingModel(
            config=config,
            model="mxbai-embed-large",
            base_url="http://remote:8080",
        )

        kw = mock_ollama_embeddings.call_args.kwargs
        assert kw["model"] == "mxbai-embed-large"
        assert kw["base_url"] == "http://remote:8080"

    def test_init_removes_api_key_param(self, mock_ollama_embeddings):
        """Test that api_key is removed before client init."""
        OllamaEmbeddingModel(
            api_key="some-secret", model="nomic-embed-text"
        )

        kw = mock_ollama_embeddings.call_args.kwargs
        assert "api_key" not in kw

    def test_register_class_loading_registers_in_factory(self):
        """Test that the class is registered in the factory."""
        from src.domain.embedding.types import EmbeddingProvider
        from src.infrastructure.embedding.factory import EmbeddingFactory

        assert EmbeddingProvider.HUGGINGFACE in EmbeddingFactory._registry
        assert (
            EmbeddingFactory._registry[EmbeddingProvider.HUGGINGFACE]
            == OllamaEmbeddingModel
        )

    def test_embed_delegates_to_client(self, mock_ollama_embeddings):
        """Test that embed() calls embed_documents and returns DTO."""
        mock_inst = mock_ollama_embeddings.return_value
        mock_inst.embed_documents.return_value = [[0.1, 0.2]]

        model = OllamaEmbeddingModel(model="nomic-embed-text")
        splitter = MagicMock(
            chunks=["hello"],
            chunk_id=["c1"],
            document_path="/t.txt",
            split_method="char",
        )
        result = model.embed(splitter)

        mock_inst.embed_documents.assert_called_once_with(["hello"])
        assert result.embeddings == [[0.1, 0.2]]
