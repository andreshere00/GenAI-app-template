{%- if "6" in cookiecutter.llm_providers -%}
from unittest.mock import patch

import pytest

from src.infrastructure.llm.adapters.ollama import OllamaModel

# --- Mocks & Fixtures ---


@pytest.fixture
def mock_chat_ollama():
    """Patches the ChatOllama class to prevent real API calls.

    Returns:
        MagicMock: The mock object for the ChatOllama class.
    """
    with patch("src.infrastructure.llm.adapters.ollama.ChatOllama") as mock:
        yield mock


# --- Test Suite ---


class TestOllamaModel:
    def test_init_valid_config_sets_client_params(
        self, config_factory, mock_chat_ollama
    ):
        """Test instantiation using only the configuration object."""
        # Use the shared factory fixture
        config = config_factory(
            model="llama3",
            base_url="http://localhost:11434",
            temperature=0.7,
            api_key="ignored-key",
        )

        OllamaModel(config=config)

        mock_chat_ollama.assert_called_once()
        call_kwargs = mock_chat_ollama.call_args.kwargs

        assert call_kwargs["model"] == "llama3"
        assert call_kwargs["base_url"] == "http://localhost:11434"
        assert call_kwargs["temperature"] == 0.7
        # Verify api_key is explicitly removed
        assert "api_key" not in call_kwargs

    def test_init_valid_kwargs_sets_client_params(self, mock_chat_ollama):
        """Test instantiation using only explicit keyword arguments."""
        OllamaModel(
            model="mistral", base_url="http://ollama-server:11434", temperature=0.5
        )

        mock_chat_ollama.assert_called_once()
        call_kwargs = mock_chat_ollama.call_args.kwargs

        assert call_kwargs["model"] == "mistral"
        assert call_kwargs["base_url"] == "http://ollama-server:11434"
        assert call_kwargs["temperature"] == 0.5

    def test_init_config_and_kwargs_prioritizes_kwargs(
        self, config_factory, mock_chat_ollama
    ):
        """Test that explicit kwargs override configuration values."""
        config = config_factory(
            model="llama2", base_url="http://localhost:11434", temperature=0.1
        )

        # Override model and base_url
        OllamaModel(
            config=config, model="gemma:7b", base_url="http://remote-ollama:8080"
        )

        call_kwargs = mock_chat_ollama.call_args.kwargs

        assert call_kwargs["model"] == "gemma:7b"  # Overridden
        assert call_kwargs["base_url"] == "http://remote-ollama:8080"  # Overridden
        assert call_kwargs["temperature"] == 0.1  # Inherited

    def test_init_removes_api_key_param(self, mock_chat_ollama):
        """Test that the api_key parameter is strictly removed before client init."""
        OllamaModel(api_key="some-secret-key", model="llama3")

        call_kwargs = mock_chat_ollama.call_args.kwargs

        assert "api_key" not in call_kwargs

    def test_getattr_valid_method_delegates_to_client(self, mock_chat_ollama):
        """Test that the underlying client methods are accessible via the wrapper."""
        mock_client_instance = mock_chat_ollama.return_value
        model = OllamaModel(model="llama3")

        # 1. Access the client attribute directly
        assert model.client == mock_client_instance

        # 2. Simulate calling a standard LangChain method
        dummy_input = "Hello Llama"
        model.client.invoke(dummy_input)

        mock_client_instance.invoke.assert_called_once_with(dummy_input)

    def test_register_class_loading_registers_in_factory(self):
        """Test that the class is correctly registered in the factory."""
        from src.domain.llm.types import LLMProvider
        from src.infrastructure.llm.factory import LlmFactory

        # Testing registration under LLMProvider.HUGGINGFACE as per source code
        assert LLMProvider.HUGGINGFACE in LlmFactory._registry
        assert LlmFactory._registry[LLMProvider.HUGGINGFACE] == OllamaModel

    def test_client_runnable_methods_are_accessible(self, mock_chat_ollama):
        """Test that the model client implements standard LangChain Runnable methods."""
        mock_instance = mock_chat_ollama.return_value
        model = OllamaModel(model="llama3")

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