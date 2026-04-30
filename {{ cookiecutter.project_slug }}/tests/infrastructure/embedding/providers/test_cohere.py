{%- if "3" in cookiecutter.embedding_providers -%}
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from src.infrastructure.embedding.adapters.cohere import (
    CohereEmbeddingModel,
)

# ---- Mocks, fixtures & helpers ---- #


@pytest.fixture
def mock_cohere_embeddings():
    """Patches CohereEmbeddings to prevent real API calls."""
    with patch(
        "src.infrastructure.embedding.adapters"
        ".cohere.CohereEmbeddings"
    ) as mock:
        yield mock


# ---- Happy path ---- #


class TestCohereEmbeddingModel:
    def test_init_valid_config_sets_client_params(
        self, config_factory, mock_cohere_embeddings
    ):
        """Test instantiation using only the configuration object."""
        config = config_factory(
            api_key="sk-cohere", model="embed-english-v3.0"
        )

        CohereEmbeddingModel(config=config)

        mock_cohere_embeddings.assert_called_once()
        kw = mock_cohere_embeddings.call_args.kwargs
        assert kw["cohere_api_key"] == "sk-cohere"
        assert kw["model"] == "embed-english-v3.0"

    def test_init_valid_kwargs_sets_client_params(
        self, mock_cohere_embeddings
    ):
        """Test instantiation using only explicit keyword arguments."""
        CohereEmbeddingModel(
            api_key="sk-kwarg", model="embed-multilingual-v3.0"
        )

        kw = mock_cohere_embeddings.call_args.kwargs
        assert kw["cohere_api_key"] == "sk-kwarg"
        assert kw["model"] == "embed-multilingual-v3.0"

    def test_init_config_and_kwargs_prioritizes_kwargs(
        self, config_factory, mock_cohere_embeddings
    ):
        """Test that explicit kwargs override configuration values."""
        config = config_factory(
            api_key="sk-cfg", model="embed-english-v3.0"
        )

        CohereEmbeddingModel(
            config=config, model="embed-english-light-v3.0"
        )

        kw = mock_cohere_embeddings.call_args.kwargs
        assert kw["cohere_api_key"] == "sk-cfg"
        assert kw["model"] == "embed-english-light-v3.0"

    def test_init_remappable_keys_maps_to_langchain_params(
        self, mock_cohere_embeddings
    ):
        """Test that api_key is mapped to cohere_api_key."""
        CohereEmbeddingModel(api_key="sk-123")

        kw = mock_cohere_embeddings.call_args.kwargs
        assert "api_key" not in kw
        assert kw["cohere_api_key"] == "sk-123"

    def test_embed_delegates_to_client(self, mock_cohere_embeddings):
        """Test that embed() calls embed_documents and returns DTO."""
        mock_inst = mock_cohere_embeddings.return_value
        mock_inst.embed_documents.return_value = [[0.1, 0.2]]

        model = CohereEmbeddingModel(api_key="sk-test")
        splitter = MagicMock(
            chunks=["hello"],
            chunk_id=["c1"],
            document_path="/t.txt",
            split_method="char",
        )
        result = model.embed(splitter)

        mock_inst.embed_documents.assert_called_once_with(["hello"])
        assert result.embeddings == [[0.1, 0.2]]

    def test_register_class_loading_registers_in_factory(self):
        """Test that the class is registered in the factory."""
        from src.domain.embedding.types import EmbeddingProvider
        from src.infrastructure.embedding.factory import EmbeddingFactory

        assert EmbeddingProvider.COHERE in EmbeddingFactory._registry
        assert (
            EmbeddingFactory._registry[EmbeddingProvider.COHERE]
            == CohereEmbeddingModel
        )
{%- endif -%}
