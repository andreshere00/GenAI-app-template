{%- if "8" in cookiecutter.embedding_providers -%}
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from src.infrastructure.embedding.adapters.voyageai import (
    VoyageAIEmbeddingModel,
)

# ---- Mocks, fixtures & helpers ---- #


@pytest.fixture
def mock_voyage_embeddings():
    """Patches VoyageAIEmbeddings to prevent real API calls."""
    with patch(
        "src.infrastructure.embedding.adapters"
        ".voyageai.VoyageAIEmbeddings"
    ) as mock:
        yield mock


# ---- Happy path ---- #


class TestVoyageAIEmbeddingModel:
    def test_init_valid_config_sets_client_params(
        self, config_factory, mock_voyage_embeddings
    ):
        """Test instantiation using only the configuration object."""
        config = config_factory(api_key="sk-voyage", model="voyage-3")

        VoyageAIEmbeddingModel(config=config)

        mock_voyage_embeddings.assert_called_once()
        kw = mock_voyage_embeddings.call_args.kwargs
        assert kw["voyage_api_key"] == "sk-voyage"
        assert kw["model"] == "voyage-3"

    def test_init_valid_kwargs_sets_client_params(
        self, mock_voyage_embeddings
    ):
        """Test instantiation using only explicit keyword arguments."""
        VoyageAIEmbeddingModel(
            api_key="sk-kwarg", model="voyage-code-3"
        )

        kw = mock_voyage_embeddings.call_args.kwargs
        assert kw["voyage_api_key"] == "sk-kwarg"
        assert kw["model"] == "voyage-code-3"

    def test_init_config_and_kwargs_prioritizes_kwargs(
        self, config_factory, mock_voyage_embeddings
    ):
        """Test that explicit kwargs override configuration values."""
        config = config_factory(api_key="sk-cfg", model="voyage-3")

        VoyageAIEmbeddingModel(config=config, model="voyage-3-lite")

        kw = mock_voyage_embeddings.call_args.kwargs
        assert kw["voyage_api_key"] == "sk-cfg"
        assert kw["model"] == "voyage-3-lite"

    def test_init_remappable_keys_maps_to_langchain_params(
        self, mock_voyage_embeddings
    ):
        """Test that api_key is mapped to voyage_api_key."""
        VoyageAIEmbeddingModel(api_key="sk-123")

        kw = mock_voyage_embeddings.call_args.kwargs
        assert "api_key" not in kw
        assert kw["voyage_api_key"] == "sk-123"

    def test_embed_delegates_to_client(self, mock_voyage_embeddings):
        """Test that embed() calls embed_documents and returns DTO."""
        mock_inst = mock_voyage_embeddings.return_value
        mock_inst.embed_documents.return_value = [[0.1, 0.2]]

        model = VoyageAIEmbeddingModel(api_key="sk-test")
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

        assert EmbeddingProvider.VOYAGEAI in EmbeddingFactory._registry
        assert (
            EmbeddingFactory._registry[EmbeddingProvider.VOYAGEAI]
            == VoyageAIEmbeddingModel
        )
{%- endif -%}
