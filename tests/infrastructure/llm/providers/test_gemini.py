from unittest.mock import patch

import pytest
from pydantic import SecretStr

from src.infrastructure.llm.adapters.gemini import GeminiModel

# --- Mocks & Fixtures ---


@pytest.fixture
def mock_chat_google():
    """Patches the ChatGoogleGenerativeAI class to prevent real API calls.

    Returns:
        MagicMock: The mock object for the ChatGoogleGenerativeAI class.
    """
    with patch("src.infrastructure.llm.adapters.gemini.ChatGoogleGenerativeAI") as mock:
        yield mock


@pytest.fixture(autouse=True)
def mock_param_map():
    """Patches the parameter map to ensure mapping logic is tested consistently.

    This ensures tests pass regardless of the actual content of src.domain.llm.constants.
    """
    test_map = {
        "api_key": "google_api_key",
        "model": "model",  # Explicit mapping if needed, or identity
    }
    with patch.dict("src.domain.llm.constants.GEMINI_PARAM_MAP", test_map, clear=True):
        yield


# --- Test Suite ---


class TestGeminiModel:

    def test_init_valid_config_sets_client_params(self, config_factory, mock_chat_google):
        """Test instantiation using only the configuration object."""
        # Use the shared factory fixture
        config = config_factory(api_key="sk-gemini-config", model="gemini-1.5-pro", temperature=0.7)

        GeminiModel(config=config)

        mock_chat_google.assert_called_once()
        call_kwargs = mock_chat_google.call_args.kwargs

        assert call_kwargs["google_api_key"] == "sk-gemini-config"
        assert call_kwargs["model"] == "gemini-1.5-pro"
        assert call_kwargs["temperature"] == 0.7

    def test_init_valid_kwargs_sets_client_params(self, mock_chat_google):
        """Test instantiation using only explicit keyword arguments."""
        GeminiModel(api_key="sk-gemini-kwarg", model="gemini-pro-vision", temperature=0.5)

        mock_chat_google.assert_called_once()
        call_kwargs = mock_chat_google.call_args.kwargs

        assert call_kwargs["google_api_key"] == "sk-gemini-kwarg"
        assert call_kwargs["model"] == "gemini-pro-vision"
        assert call_kwargs["temperature"] == 0.5

    def test_init_config_and_kwargs_prioritizes_kwargs(self, config_factory, mock_chat_google):
        """Test that explicit kwargs override configuration values."""
        config = config_factory(api_key="sk-gemini-config", model="gemini-1.0-pro", temperature=0.1)

        # Override model and temperature, keep api_key from config
        GeminiModel(config=config, model="gemini-1.5-flash", temperature=0.9)

        call_kwargs = mock_chat_google.call_args.kwargs

        assert call_kwargs["google_api_key"] == "sk-gemini-config"  # Inherited
        assert call_kwargs["model"] == "gemini-1.5-flash"  # Overridden
        assert call_kwargs["temperature"] == 0.9  # Overridden

    def test_init_remappable_keys_maps_to_langchain_params(self, mock_chat_google):
        """Test that generic keys are correctly mapped to LangChain Google keys."""
        GeminiModel(api_key="sk-gemini-123")

        call_kwargs = mock_chat_google.call_args.kwargs

        # Ensure original keys are removed and mapped keys are present
        assert "api_key" not in call_kwargs
        assert call_kwargs["google_api_key"] == "sk-gemini-123"

    def test_init_secret_str_api_key_passes_raw_secret(self, mock_chat_google):
        """Test that Pydantic SecretStr is handled correctly."""
        secret_key = SecretStr("sk-gemini-secret")
        GeminiModel(api_key=secret_key)

        call_kwargs = mock_chat_google.call_args.kwargs
        assert call_kwargs["google_api_key"] == secret_key

    def test_getattr_valid_method_delegates_to_client(self, mock_chat_google):
        """Test that the underlying client methods are accessible via the wrapper."""
        mock_client_instance = mock_chat_google.return_value
        model = GeminiModel(api_key="sk-test")

        # 1. Access the client attribute directly
        assert model.client == mock_client_instance

        # 2. Simulate calling a standard LangChain method (e.g., invoke)
        dummy_input = "Hello Gemini"
        model.client.invoke(dummy_input)

        mock_client_instance.invoke.assert_called_once_with(dummy_input)

    def test_register_class_loading_registers_in_factory(self):
        """Test that the class is correctly registered in the factory."""
        from src.domain.llm.types import LLMProvider
        from src.infrastructure.llm.factory import LlmFactory

        assert LLMProvider.GOOGLE in LlmFactory._registry
        assert LlmFactory._registry[LLMProvider.GOOGLE] == GeminiModel

    def test_client_runnable_methods_are_accessible(self, mock_chat_google):
        """Test that the model client implements standard LangChain Runnable methods."""
        mock_instance = mock_chat_google.return_value
        model = GeminiModel(api_key="sk-test")

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
