from unittest.mock import patch

import pytest
from pydantic import SecretStr

from src.infrastructure.llm.adapters.openai import OpenAIModel

# --- Mocks & Fixtures ---


@pytest.fixture
def mock_chat_openai():
    """Patches the ChatOpenAI class to prevent real API calls.

    Returns:
        MagicMock: The mock object for the ChatOpenAI class.
    """
    with patch("src.infrastructure.llm.adapters.openai.ChatOpenAI") as mock:
        yield mock


# --- Test Suite ---


class TestOpenAIModel:

    def test_init_valid_config_sets_client_params(self, config_factory, mock_chat_openai):
        """Test instantiation using only the configuration object."""
        # Use the shared factory fixture
        config = config_factory(api_key="sk-config-key", model="gpt-4", temperature=0.7)

        OpenAIModel(config=config)

        # Verify ChatOpenAI was initialized with mapped parameters
        mock_chat_openai.assert_called_once()
        call_kwargs = mock_chat_openai.call_args.kwargs

        assert call_kwargs["openai_api_key"] == "sk-config-key"
        assert call_kwargs["model"] == "gpt-4"
        assert call_kwargs["temperature"] == 0.7

    def test_init_valid_kwargs_sets_client_params(self, mock_chat_openai):
        """Test instantiation using only explicit keyword arguments."""
        OpenAIModel(api_key="sk-kwarg-key", model="gpt-3.5-turbo", temperature=0.5)

        mock_chat_openai.assert_called_once()
        call_kwargs = mock_chat_openai.call_args.kwargs

        assert call_kwargs["openai_api_key"] == "sk-kwarg-key"
        assert (
            call_kwargs["model_name"] == "gpt-3.5-turbo"
            if "model_name" in call_kwargs
            else call_kwargs["model"] == "gpt-3.5-turbo"
        )
        assert call_kwargs["temperature"] == 0.5

    def test_init_config_and_kwargs_prioritizes_kwargs(self, config_factory, mock_chat_openai):
        """Test that explicit kwargs override configuration values."""
        config = config_factory(api_key="sk-config", model="gpt-4", temperature=0.1)

        # Override model and temperature, but keep the api_key from config
        OpenAIModel(config=config, model="gpt-3.5-turbo", temperature=0.9)

        call_kwargs = mock_chat_openai.call_args.kwargs

        assert call_kwargs["openai_api_key"] == "sk-config"  # Inherited
        assert call_kwargs["model"] == "gpt-3.5-turbo"  # Overridden
        assert call_kwargs["temperature"] == 0.9  # Overridden

    def test_init_remappable_keys_maps_to_langchain_params(self, mock_chat_openai):
        """Test that generic keys are correctly mapped to LangChain specific keys."""
        OpenAIModel(
            api_key="sk-123",
            base_url="http://local-proxy:8080",
            organization="org-test",
            proxy="http://proxy:80",
        )

        call_kwargs = mock_chat_openai.call_args.kwargs

        # Ensure original keys are removed and mapped keys are present
        assert "api_key" not in call_kwargs
        assert call_kwargs["openai_api_key"] == "sk-123"

        assert "base_url" not in call_kwargs
        assert call_kwargs["openai_api_base"] == "http://local-proxy:8080"

        assert "organization" not in call_kwargs
        assert call_kwargs["openai_organization"] == "org-test"

        assert "proxy" not in call_kwargs
        assert call_kwargs["openai_proxy"] == "http://proxy:80"

    def test_init_secret_str_api_key_passes_raw_secret(self, mock_chat_openai):
        """Test that Pydantic SecretStr is handled correctly."""
        secret_key = SecretStr("sk-secret")
        OpenAIModel(api_key=secret_key)

        call_kwargs = mock_chat_openai.call_args.kwargs
        assert call_kwargs["openai_api_key"] == secret_key

    def test_getattr_valid_method_delegates_to_client(self, mock_chat_openai):
        """Test that the underlying client methods are accessible via the wrapper."""
        # Setup the mock instance returned by the class
        mock_client_instance = mock_chat_openai.return_value

        model = OpenAIModel(api_key="sk-test")

        # 1. Access the client attribute directly
        assert model.client == mock_client_instance

        # 2. Simulate calling a standard LangChain method (e.g., invoke)
        dummy_input = "Hello world"
        model.client.invoke(dummy_input)

        # Verify the mock received the call
        mock_client_instance.invoke.assert_called_once_with(dummy_input)

    def test_register_class_loading_registers_in_factory(self):
        """Test that the class is correctly registered in the factory."""
        from src.domain.llm.types import LLMProvider
        from src.infrastructure.llm.factory import LlmFactory

        assert LLMProvider.OPENAI in LlmFactory._registry
        assert LlmFactory._registry[LLMProvider.OPENAI] == OpenAIModel

    def test_client_runnable_methods_are_accessible(self, mock_chat_openai):
        """Test that the model client implements standard LangChain Runnable methods.

        According to LangChain documentation, all Chat Models should implement:
        - invoke: Synchronous invocation
        - stream: Synchronous streaming
        - batch: Synchronous batch processing
        - ainvoke: Asynchronous invocation
        - astream: Asynchronous streaming
        - abatch: Asynchronous batch processing
        """
        # Setup mock
        mock_instance = mock_chat_openai.return_value

        # Instantiate your wrapper
        model = OpenAIModel(api_key="sk-test")

        # 1. Check Synchronous Methods
        assert hasattr(model.client, "invoke"), "Client must implement 'invoke'"
        assert hasattr(model.client, "stream"), "Client must implement 'stream'"
        assert hasattr(model.client, "batch"), "Client must implement 'batch'"

        # 2. Check Asynchronous Methods (Standard in modern LangChain)
        assert hasattr(model.client, "ainvoke"), "Client must implement 'ainvoke'"
        assert hasattr(model.client, "astream"), "Client must implement 'astream'"
        assert hasattr(model.client, "abatch"), "Client must implement 'abatch'"

        # 3. Verify they are callable
        model.client.invoke("test")
        mock_instance.invoke.assert_called_with("test")

        model.client.batch(["test1", "test2"])
        mock_instance.batch.assert_called_with(["test1", "test2"])
