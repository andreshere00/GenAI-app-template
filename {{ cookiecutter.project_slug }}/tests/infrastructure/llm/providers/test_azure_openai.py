{%- if "2" in cookiecutter.llm_providers -%}
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from src.infrastructure.llm.adapters.azure_openai import AzureOpenAIModel

# --- Mocks & Fixtures ---


@pytest.fixture
def mock_chat_azure():
    """Patches the AzureChatOpenAI class to prevent real API calls.

    Returns:
        MagicMock: The mock object for the AzureChatOpenAI class.
    """
    with patch("src.infrastructure.llm.adapters.azure_openai.AzureChatOpenAI") as mock:
        yield mock


@pytest.fixture(autouse=True)
def mock_param_map():
    """Patches the parameter map to ensure mapping logic is tested consistently.

    This maps generic config keys to Azure-specific keys.
    """
    test_map = {
        "base_url": "azure_endpoint",
        "model": "azure_deployment",
        "api_key": "api_key",
    }
    with patch.dict(
        "src.domain.llm.constants.AZURE_OPENAI_PARAM_MAP", test_map, clear=True
    ):
        yield


# --- Test Suite ---


class TestAzureOpenAIModel:
    def test_init_valid_config_sets_client_params(
        self, config_factory, mock_chat_azure
    ):
        """Test instantiation using only the configuration object."""
        config = config_factory(
            api_key="sk-azure-key",
            model="gpt-4-deployment",
            base_url="https://my-resource.openai.azure.com/",
            api_version="2023-05-15",
            temperature=0.7,
        )

        AzureOpenAIModel(config=config)

        mock_chat_azure.assert_called_once()
        call_kwargs = mock_chat_azure.call_args.kwargs

        # Verify mappings based on mock_param_map
        assert call_kwargs["api_key"] == "sk-azure-key"
        assert call_kwargs["azure_deployment"] == "gpt-4-deployment"
        assert call_kwargs["azure_endpoint"] == "https://my-resource.openai.azure.com/"
        assert call_kwargs["api_version"] == "2023-05-15"
        assert call_kwargs["temperature"] == 0.7

    def test_init_valid_kwargs_sets_client_params(self, mock_chat_azure):
        """Test instantiation using only explicit keyword arguments."""
        AzureOpenAIModel(
            api_key="sk-kwarg-key",
            azure_endpoint="https://kwarg-resource.com",
            azure_deployment="kwarg-deployment",
            api_version="2023-12-01",
        )

        mock_chat_azure.assert_called_once()
        call_kwargs = mock_chat_azure.call_args.kwargs

        assert call_kwargs["api_key"] == "sk-kwarg-key"
        assert call_kwargs["azure_endpoint"] == "https://kwarg-resource.com"
        assert call_kwargs["azure_deployment"] == "kwarg-deployment"
        assert call_kwargs["api_version"] == "2023-12-01"

    def test_init_config_and_kwargs_prioritizes_kwargs(
        self, config_factory, mock_chat_azure
    ):
        """Test that explicit kwargs override configuration values."""
        config = config_factory(
            api_key="sk-config",
            model="config-deployment",
            base_url="https://config.com",
            temperature=0.1,
        )

        # Override deployment and endpoint via kwargs
        AzureOpenAIModel(
            config=config,
            model="override-deployment",
            azure_endpoint="https://override.com",
            temperature=0.9,
        )

        call_kwargs = mock_chat_azure.call_args.kwargs

        assert call_kwargs["api_key"] == "sk-config"  # Inherited
        assert call_kwargs["azure_deployment"] == "override-deployment"  # Overridden
        assert call_kwargs["azure_endpoint"] == "https://override.com"  # Overridden
        assert call_kwargs["temperature"] == 0.9  # Overridden

    def test_init_remappable_keys_maps_to_langchain_params(self, mock_chat_azure):
        """Test that generic keys are correctly mapped to Azure specific keys."""
        # We pass 'base_url' and 'model' as kwargs, expecting them to be mapped
        AzureOpenAIModel(
            base_url="https://mapped-endpoint.com",
            model="mapped-deployment",
            api_key="sk-test",
        )

        call_kwargs = mock_chat_azure.call_args.kwargs

        # Ensure original keys are removed and mapped keys are present
        assert "base_url" not in call_kwargs
        assert "model" not in call_kwargs

        assert call_kwargs["azure_endpoint"] == "https://mapped-endpoint.com"
        assert call_kwargs["azure_deployment"] == "mapped-deployment"

    def test_init_secret_str_api_key_passes_raw_secret(self, mock_chat_azure):
        """Test that Pydantic SecretStr is handled correctly."""
        secret_key = SecretStr("sk-azure-secret")
        AzureOpenAIModel(api_key=secret_key)

        call_kwargs = mock_chat_azure.call_args.kwargs
        assert call_kwargs["api_key"] == secret_key

    def test_getattr_valid_method_delegates_to_client(self, mock_chat_azure):
        """Test that the underlying client methods are accessible via the wrapper."""
        mock_client_instance = mock_chat_azure.return_value
        model = AzureOpenAIModel(api_key="sk-test")

        # 1. Access the client attribute directly
        assert model.client == mock_client_instance

        # 2. Simulate calling a standard LangChain method
        dummy_input = "Hello Azure"
        model.client.invoke(dummy_input)

        mock_client_instance.invoke.assert_called_once_with(dummy_input)

    def test_register_class_loading_registers_in_factory(self):
        """Test that the class is correctly registered in the factory."""
        from src.domain.llm.types import LLMProvider
        from src.infrastructure.llm.factory import LlmFactory

        assert LLMProvider.AZURE in LlmFactory._registry
        assert LlmFactory._registry[LLMProvider.AZURE] == AzureOpenAIModel

    def test_client_runnable_methods_are_accessible(self, mock_chat_azure):
        """Test that the model client implements standard LangChain Runnable methods."""
        mock_instance = mock_chat_azure.return_value
        model = AzureOpenAIModel(api_key="sk-test")

        # 1. Check Synchronous Methods
        assert hasattr(model.client, "invoke"), "Client must implement 'invoke'"
        assert hasattr(model.client, "stream"), "Client must implement 'stream'"
        assert hasattr(model.client, "batch"), "Client must implement 'batch'"

        # 2. Check Asynchronous Methods
        assert hasattr(model.client, "ainvoke"), "Client must implement 'ainvoke'"
        assert hasattr(model.client, "astream"), "Client must implement 'astream'"
        assert hasattr(model.client, "abatch"), "Client must implement 'abatch'"

        # 3. Verify they are callable
        model.client.invoke("test")
        mock_instance.invoke.assert_called_with("test")
{%- endif -%}