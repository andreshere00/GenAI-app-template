from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from src.infrastructure.embedding.adapters.grok import (
    GrokEmbeddingModel,
    XAI_BASE_URL,
)

# ---- Mocks, fixtures & helpers ---- #


@pytest.fixture
def mock_openai_embeddings():
    """Patches OpenAIEmbeddings used by Grok adapter."""
    with patch(
        "src.infrastructure.embedding.adapters.grok.OpenAIEmbeddings"
    ) as mock:
        yield mock


# ---- Happy path ---- #


class TestGrokEmbeddingModel:
    def test_init_valid_config_sets_client_params(
        self, config_factory, mock_openai_embeddings
    ):
        """Test instantiation using only the configuration object."""
        config = config_factory(api_key="sk-xai", model="grok-embed")

        GrokEmbeddingModel(config=config)

        mock_openai_embeddings.assert_called_once()
        kw = mock_openai_embeddings.call_args.kwargs
        assert kw["openai_api_key"] == "sk-xai"
        assert kw["model"] == "grok-embed"

    def test_init_valid_kwargs_sets_client_params(
        self, mock_openai_embeddings
    ):
        """Test instantiation using only explicit keyword arguments."""
        GrokEmbeddingModel(api_key="sk-kwarg", model="grok-embed-v2")

        kw = mock_openai_embeddings.call_args.kwargs
        assert kw["openai_api_key"] == "sk-kwarg"
        assert kw["model"] == "grok-embed-v2"

    def test_init_config_and_kwargs_prioritizes_kwargs(
        self, config_factory, mock_openai_embeddings
    ):
        """Test that explicit kwargs override configuration values."""
        config = config_factory(api_key="sk-cfg", model="grok-embed")

        GrokEmbeddingModel(config=config, model="grok-embed-v2")

        kw = mock_openai_embeddings.call_args.kwargs
        assert kw["openai_api_key"] == "sk-cfg"
        assert kw["model"] == "grok-embed-v2"

    def test_init_sets_default_xai_base_url(
        self, mock_openai_embeddings
    ):
        """Test that the xAI base URL is set by default."""
        GrokEmbeddingModel(api_key="sk-test")

        kw = mock_openai_embeddings.call_args.kwargs
        assert kw["openai_api_base"] == XAI_BASE_URL

    def test_init_custom_base_url_overrides_default(
        self, mock_openai_embeddings
    ):
        """Test that an explicit base_url overrides the xAI default."""
        GrokEmbeddingModel(
            api_key="sk-test", base_url="http://custom:8080"
        )

        kw = mock_openai_embeddings.call_args.kwargs
        assert kw["openai_api_base"] == "http://custom:8080"

    def test_init_remappable_keys_maps_to_langchain_params(
        self, mock_openai_embeddings
    ):
        """Test that api_key and base_url are mapped to OpenAI keys."""
        GrokEmbeddingModel(api_key="sk-123")

        kw = mock_openai_embeddings.call_args.kwargs
        assert "api_key" not in kw
        assert kw["openai_api_key"] == "sk-123"

    def test_init_secret_str_api_key_passes_raw(
        self, mock_openai_embeddings
    ):
        """Test that Pydantic SecretStr is handled correctly."""
        secret = SecretStr("sk-secret")
        GrokEmbeddingModel(api_key=secret)

        kw = mock_openai_embeddings.call_args.kwargs
        assert kw["openai_api_key"] == secret

    def test_register_class_loading_registers_in_factory(self):
        """Test that the class is registered in the factory."""
        from src.domain.embedding.types import EmbeddingProvider
        from src.infrastructure.embedding.factory import EmbeddingFactory

        assert EmbeddingProvider.XAI in EmbeddingFactory._registry
        assert (
            EmbeddingFactory._registry[EmbeddingProvider.XAI]
            == GrokEmbeddingModel
        )

    def test_embed_delegates_to_client(self, mock_openai_embeddings):
        """Test that embed() calls embed_documents and returns DTO."""
        mock_inst = mock_openai_embeddings.return_value
        mock_inst.embed_documents.return_value = [[0.1, 0.2]]

        model = GrokEmbeddingModel(api_key="sk-test")
        splitter = MagicMock(
            chunks=["hello"],
            chunk_id=["c1"],
            document_path="/t.txt",
            split_method="char",
        )
        result = model.embed(splitter)

        mock_inst.embed_documents.assert_called_once_with(["hello"])
        assert result.embeddings == [[0.1, 0.2]]
