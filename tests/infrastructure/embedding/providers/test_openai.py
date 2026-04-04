from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from src.infrastructure.embedding.adapters.openai import (
    OpenAIEmbeddingModel,
)

# ---- Mocks, fixtures & helpers ---- #


@pytest.fixture
def mock_openai_embeddings():
    """Patches OpenAIEmbeddings to prevent real API calls."""
    with patch(
        "src.infrastructure.embedding.adapters.openai.OpenAIEmbeddings"
    ) as mock:
        yield mock


# ---- Happy path ---- #


class TestOpenAIEmbeddingModel:
    def test_init_valid_config_sets_client_params(
        self, config_factory, mock_openai_embeddings
    ):
        """Test instantiation using only the configuration object."""
        config = config_factory(
            api_key="sk-config", model="text-embedding-3-small"
        )

        OpenAIEmbeddingModel(config=config)

        mock_openai_embeddings.assert_called_once()
        kw = mock_openai_embeddings.call_args.kwargs
        assert kw["openai_api_key"] == "sk-config"
        assert kw["model"] == "text-embedding-3-small"

    def test_init_valid_kwargs_sets_client_params(
        self, mock_openai_embeddings
    ):
        """Test instantiation using only explicit keyword arguments."""
        OpenAIEmbeddingModel(
            api_key="sk-kwarg", model="text-embedding-ada-002"
        )

        kw = mock_openai_embeddings.call_args.kwargs
        assert kw["openai_api_key"] == "sk-kwarg"
        assert kw["model"] == "text-embedding-ada-002"

    def test_init_config_and_kwargs_prioritizes_kwargs(
        self, config_factory, mock_openai_embeddings
    ):
        """Test that explicit kwargs override configuration values."""
        config = config_factory(
            api_key="sk-config", model="text-embedding-3-small"
        )

        OpenAIEmbeddingModel(
            config=config, model="text-embedding-3-large"
        )

        kw = mock_openai_embeddings.call_args.kwargs
        assert kw["openai_api_key"] == "sk-config"
        assert kw["model"] == "text-embedding-3-large"

    def test_init_remappable_keys_maps_to_langchain_params(
        self, mock_openai_embeddings
    ):
        """Test that generic keys are mapped to LangChain specific keys."""
        OpenAIEmbeddingModel(
            api_key="sk-123",
            base_url="http://proxy:8080",
            organization="org-test",
        )

        kw = mock_openai_embeddings.call_args.kwargs
        assert "api_key" not in kw
        assert kw["openai_api_key"] == "sk-123"
        assert kw["openai_api_base"] == "http://proxy:8080"
        assert kw["openai_organization"] == "org-test"

    def test_init_secret_str_api_key_passes_raw(
        self, mock_openai_embeddings
    ):
        """Test that Pydantic SecretStr is handled correctly."""
        secret = SecretStr("sk-secret")
        OpenAIEmbeddingModel(api_key=secret)

        kw = mock_openai_embeddings.call_args.kwargs
        assert kw["openai_api_key"] == secret

    def test_register_class_loading_registers_in_factory(self):
        """Test that the class is registered in the factory."""
        from src.domain.embedding.types import EmbeddingProvider
        from src.infrastructure.embedding.factory import EmbeddingFactory

        assert EmbeddingProvider.OPENAI in EmbeddingFactory._registry
        assert (
            EmbeddingFactory._registry[EmbeddingProvider.OPENAI]
            == OpenAIEmbeddingModel
        )

    def test_embed_delegates_to_client(self, mock_openai_embeddings):
        """Test that embed() calls embed_documents and returns DTO."""
        mock_inst = mock_openai_embeddings.return_value
        mock_inst.embed_documents.return_value = [[0.1, 0.2]]

        model = OpenAIEmbeddingModel(api_key="sk-test")
        splitter = MagicMock(
            chunks=["hello"],
            chunk_id=["c1"],
            document_path="/t.txt",
            split_method="char",
        )
        result = model.embed(splitter)

        mock_inst.embed_documents.assert_called_once_with(["hello"])
        assert result.embeddings == [[0.1, 0.2]]
        assert result.chunks == ["hello"]

    def test_client_embed_methods_are_accessible(
        self, mock_openai_embeddings
    ):
        """Test that embedding client methods are accessible."""
        mock_inst = mock_openai_embeddings.return_value
        model = OpenAIEmbeddingModel(api_key="sk-test")

        assert model.client == mock_inst
        assert hasattr(model.client, "embed_documents")
        assert hasattr(model.client, "embed_query")
