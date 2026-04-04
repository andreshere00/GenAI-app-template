from unittest.mock import MagicMock, patch

import pytest

from src.infrastructure.embedding.adapters.bedrock import (
    BedrockEmbeddingModel,
)

# ---- Mocks, fixtures & helpers ---- #


@pytest.fixture
def mock_bedrock_embeddings():
    """Patches BedrockEmbeddings to prevent real API calls."""
    with patch(
        "src.infrastructure.embedding.adapters"
        ".bedrock.BedrockEmbeddings"
    ) as mock:
        yield mock


# ---- Happy path ---- #


class TestBedrockEmbeddingModel:
    def test_init_valid_config_sets_client_params(
        self, config_factory, mock_bedrock_embeddings
    ):
        """Test instantiation using only the configuration object."""
        config = config_factory(
            model="amazon.titan-embed-text-v1",
            api_key="ignored-key",
        )

        BedrockEmbeddingModel(config=config)

        mock_bedrock_embeddings.assert_called_once()
        kw = mock_bedrock_embeddings.call_args.kwargs
        assert kw["model_id"] == "amazon.titan-embed-text-v1"
        assert "api_key" not in kw

    def test_init_valid_kwargs_sets_client_params(
        self, mock_bedrock_embeddings
    ):
        """Test instantiation using only explicit keyword arguments."""
        BedrockEmbeddingModel(model="amazon.titan-embed-text-v2")

        kw = mock_bedrock_embeddings.call_args.kwargs
        assert kw["model_id"] == "amazon.titan-embed-text-v2"

    def test_init_config_and_kwargs_prioritizes_kwargs(
        self, config_factory, mock_bedrock_embeddings
    ):
        """Test that explicit kwargs override configuration values."""
        config = config_factory(model="amazon.titan-embed-text-v1")

        BedrockEmbeddingModel(
            config=config, model="cohere.embed-english-v3"
        )

        kw = mock_bedrock_embeddings.call_args.kwargs
        assert kw["model_id"] == "cohere.embed-english-v3"

    def test_init_remappable_keys_maps_to_langchain_params(
        self, mock_bedrock_embeddings
    ):
        """Test that model is mapped to model_id."""
        BedrockEmbeddingModel(model="amazon.titan-embed-text-v1")

        kw = mock_bedrock_embeddings.call_args.kwargs
        assert "model" not in kw
        assert kw["model_id"] == "amazon.titan-embed-text-v1"

    def test_init_removes_api_key_param(self, mock_bedrock_embeddings):
        """Test that api_key is removed before client init."""
        BedrockEmbeddingModel(
            api_key="some-secret", model="amazon.titan-embed-text-v1"
        )

        kw = mock_bedrock_embeddings.call_args.kwargs
        assert "api_key" not in kw

    def test_register_class_loading_registers_in_factory(self):
        """Test that the class is registered in the factory."""
        from src.domain.embedding.types import EmbeddingProvider
        from src.infrastructure.embedding.factory import EmbeddingFactory

        assert EmbeddingProvider.AWS in EmbeddingFactory._registry
        assert (
            EmbeddingFactory._registry[EmbeddingProvider.AWS]
            == BedrockEmbeddingModel
        )

    def test_embed_delegates_to_client(self, mock_bedrock_embeddings):
        """Test that embed() calls embed_documents and returns DTO."""
        mock_inst = mock_bedrock_embeddings.return_value
        mock_inst.embed_documents.return_value = [[0.1, 0.2]]

        model = BedrockEmbeddingModel(model="amazon.titan-embed-text-v1")
        splitter = MagicMock(
            chunks=["hello"],
            chunk_id=["c1"],
            document_path="/t.txt",
            split_method="char",
        )
        result = model.embed(splitter)

        mock_inst.embed_documents.assert_called_once_with(["hello"])
        assert result.embeddings == [[0.1, 0.2]]
