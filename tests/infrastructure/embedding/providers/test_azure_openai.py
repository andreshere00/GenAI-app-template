from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from src.infrastructure.embedding.adapters.azure_openai import (
    AzureOpenAIEmbeddingModel,
)

# ---- Mocks, fixtures & helpers ---- #


@pytest.fixture
def mock_azure_embeddings():
    """Patches AzureOpenAIEmbeddings to prevent real API calls."""
    with patch(
        "src.infrastructure.embedding.adapters"
        ".azure_openai.AzureOpenAIEmbeddings"
    ) as mock:
        yield mock


@pytest.fixture(autouse=True)
def mock_param_map():
    """Patches the parameter map for consistent mapping tests."""
    test_map = {
        "base_url": "azure_endpoint",
        "model": "azure_deployment",
        "api_key": "api_key",
    }
    with patch.dict(
        "src.domain.embedding.constants"
        ".AZURE_OPENAI_EMBEDDING_PARAM_MAP",
        test_map,
        clear=True,
    ):
        yield


# ---- Happy path ---- #


class TestAzureOpenAIEmbeddingModel:
    def test_init_valid_config_sets_client_params(
        self, config_factory, mock_azure_embeddings
    ):
        """Test instantiation using only the configuration object."""
        config = config_factory(
            api_key="sk-azure",
            model="embed-deploy",
            base_url="https://my.openai.azure.com/",
            api_version="2023-05-15",
        )

        AzureOpenAIEmbeddingModel(config=config)

        mock_azure_embeddings.assert_called_once()
        kw = mock_azure_embeddings.call_args.kwargs
        assert kw["api_key"] == "sk-azure"
        assert kw["azure_deployment"] == "embed-deploy"
        assert kw["azure_endpoint"] == "https://my.openai.azure.com/"
        assert kw["api_version"] == "2023-05-15"

    def test_init_valid_kwargs_sets_client_params(
        self, mock_azure_embeddings
    ):
        """Test instantiation using only explicit keyword arguments."""
        AzureOpenAIEmbeddingModel(
            api_key="sk-kwarg",
            azure_endpoint="https://kwarg.com",
            azure_deployment="kwarg-deploy",
            api_version="2023-12-01",
        )

        kw = mock_azure_embeddings.call_args.kwargs
        assert kw["api_key"] == "sk-kwarg"
        assert kw["azure_endpoint"] == "https://kwarg.com"
        assert kw["azure_deployment"] == "kwarg-deploy"

    def test_init_config_and_kwargs_prioritizes_kwargs(
        self, config_factory, mock_azure_embeddings
    ):
        """Test that explicit kwargs override configuration values."""
        config = config_factory(
            api_key="sk-cfg", model="cfg-deploy", base_url="https://cfg.com"
        )

        AzureOpenAIEmbeddingModel(
            config=config,
            model="override-deploy",
            azure_endpoint="https://override.com",
        )

        kw = mock_azure_embeddings.call_args.kwargs
        assert kw["api_key"] == "sk-cfg"
        assert kw["azure_deployment"] == "override-deploy"
        assert kw["azure_endpoint"] == "https://override.com"

    def test_init_remappable_keys_maps_to_langchain_params(
        self, mock_azure_embeddings
    ):
        """Test that generic keys are mapped to Azure-specific keys."""
        AzureOpenAIEmbeddingModel(
            base_url="https://mapped.com",
            model="mapped-deploy",
            api_key="sk-test",
        )

        kw = mock_azure_embeddings.call_args.kwargs
        assert "base_url" not in kw
        assert "model" not in kw
        assert kw["azure_endpoint"] == "https://mapped.com"
        assert kw["azure_deployment"] == "mapped-deploy"

    def test_init_secret_str_api_key_passes_raw(
        self, mock_azure_embeddings
    ):
        """Test that Pydantic SecretStr is handled correctly."""
        secret = SecretStr("sk-secret")
        AzureOpenAIEmbeddingModel(api_key=secret)

        kw = mock_azure_embeddings.call_args.kwargs
        assert kw["api_key"] == secret

    def test_register_class_loading_registers_in_factory(self):
        """Test that the class is registered in the factory."""
        from src.domain.embedding.types import EmbeddingProvider
        from src.infrastructure.embedding.factory import EmbeddingFactory

        assert EmbeddingProvider.AZURE in EmbeddingFactory._registry
        assert (
            EmbeddingFactory._registry[EmbeddingProvider.AZURE]
            == AzureOpenAIEmbeddingModel
        )

    def test_embed_delegates_to_client(self, mock_azure_embeddings):
        """Test that embed() calls embed_documents and returns DTO."""
        mock_inst = mock_azure_embeddings.return_value
        mock_inst.embed_documents.return_value = [[0.1, 0.2]]

        model = AzureOpenAIEmbeddingModel(api_key="sk-test")
        splitter = MagicMock(
            chunks=["hello"],
            chunk_id=["c1"],
            document_path="/t.txt",
            split_method="char",
        )
        result = model.embed(splitter)

        mock_inst.embed_documents.assert_called_once_with(["hello"])
        assert result.embeddings == [[0.1, 0.2]]
