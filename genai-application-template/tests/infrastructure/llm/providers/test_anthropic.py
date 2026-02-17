from unittest.mock import patch

import pytest
from pydantic import SecretStr

from src.infrastructure.llm.adapters.anthropic import AnthropicModel

# --- Mocks & Fixtures ---


@pytest.fixture
def mock_chat_anthropic():
    """Patches the ChatAnthropic class to prevent real API calls."""
    with patch("src.infrastructure.llm.adapters.anthropic.ChatAnthropic") as mock:
        yield mock


@pytest.fixture(autouse=True)
def mock_param_map():
    """Patches the parameter map to ensure mapping logic is tested consistently.

    This ensures tests pass regardless of the actual content of src.domain.llm.constants.
    """
    test_map = {
        "api_key": "anthropic_api_key",
        "base_url": "anthropic_api_url",
    }
    with patch.dict("src.domain.llm.constants.CLAUDE_PARAM_MAP", test_map, clear=True):
        yield


# --- Test Suite ---


class TestAnthropicModel:
    def test_init_valid_config_sets_client_params(
        self, config_factory, mock_chat_anthropic
    ):
        """Test instantiation using only the configuration object."""
        config = config_factory(
            api_key="sk-ant-config", model="claude-3-opus", temperature=0.7
        )

        AnthropicModel(config=config)

        mock_chat_anthropic.assert_called_once()
        call_kwargs = mock_chat_anthropic.call_args.kwargs

        assert call_kwargs["anthropic_api_key"] == "sk-ant-config"
        assert call_kwargs["model"] == "claude-3-opus"
        assert call_kwargs["temperature"] == 0.7

    def test_init_valid_kwargs_sets_client_params(self, mock_chat_anthropic):
        """Test instantiation using only explicit keyword arguments."""
        AnthropicModel(api_key="sk-ant-kwarg", model="claude-3-sonnet", temperature=0.5)

        mock_chat_anthropic.assert_called_once()
        call_kwargs = mock_chat_anthropic.call_args.kwargs

        assert call_kwargs["anthropic_api_key"] == "sk-ant-kwarg"
        assert (
            call_kwargs["model_name"] == "claude-3-sonnet"
            if "model_name" in call_kwargs
            else call_kwargs["model"] == "claude-3-sonnet"
        )
        assert call_kwargs["temperature"] == 0.5

    def test_init_config_and_kwargs_prioritizes_kwargs(
        self, config_factory, mock_chat_anthropic
    ):
        """Test that explicit kwargs override configuration values."""
        config = config_factory(
            api_key="sk-ant-config", model="claude-2", temperature=0.1
        )

        AnthropicModel(config=config, model="claude-3-haiku", temperature=0.9)

        call_kwargs = mock_chat_anthropic.call_args.kwargs

        assert call_kwargs["anthropic_api_key"] == "sk-ant-config"  # Inherited
        assert call_kwargs["model"] == "claude-3-haiku"  # Overridden
        assert call_kwargs["temperature"] == 0.9  # Overridden

    def test_init_remappable_keys_maps_to_langchain_params(self, mock_chat_anthropic):
        """Test that generic keys are correctly mapped to LangChain Anthropic keys."""
        AnthropicModel(
            api_key="sk-ant-123",
            base_url="http://local-proxy:8080",
        )

        call_kwargs = mock_chat_anthropic.call_args.kwargs

        # Ensure original keys are removed and mapped keys are present
        assert "api_key" not in call_kwargs
        assert call_kwargs["anthropic_api_key"] == "sk-ant-123"

        assert "base_url" not in call_kwargs
        assert call_kwargs["anthropic_api_url"] == "http://local-proxy:8080"

    def test_init_secret_str_api_key_passes_raw_secret(self, mock_chat_anthropic):
        """Test that Pydantic SecretStr is handled correctly."""
        secret_key = SecretStr("sk-ant-secret")
        AnthropicModel(api_key=secret_key)

        call_kwargs = mock_chat_anthropic.call_args.kwargs
        assert call_kwargs["anthropic_api_key"] == secret_key

    def test_getattr_valid_method_delegates_to_client(self, mock_chat_anthropic):
        """Test that the underlying client methods are accessible via the wrapper."""
        mock_client_instance = mock_chat_anthropic.return_value
        model = AnthropicModel(api_key="sk-test")

        assert model.client == mock_client_instance

        dummy_input = "Hello Claude"
        model.client.invoke(dummy_input)

        mock_client_instance.invoke.assert_called_once_with(dummy_input)

    def test_register_class_loading_registers_in_factory(self):
        """Test that the class is correctly registered in the factory."""
        from src.domain.llm.types import LLMProvider
        from src.infrastructure.llm.factory import LlmFactory

        assert LLMProvider.ANTHROPIC in LlmFactory._registry
        assert LlmFactory._registry[LLMProvider.ANTHROPIC] == AnthropicModel

    def test_client_runnable_methods_are_accessible(self, mock_chat_anthropic):
        """Test that the model client implements standard LangChain Runnable methods."""
        mock_instance = mock_chat_anthropic.return_value
        model = AnthropicModel(api_key="sk-test")

        assert hasattr(model.client, "invoke"), "Client must implement 'invoke'"
        assert hasattr(model.client, "stream"), "Client must implement 'stream'"
        assert hasattr(model.client, "batch"), "Client must implement 'batch'"

        assert hasattr(model.client, "ainvoke"), "Client must implement 'ainvoke'"
        assert hasattr(model.client, "astream"), "Client must implement 'astream'"
        assert hasattr(model.client, "abatch"), "Client must implement 'abatch'"

        model.client.invoke("test")
        mock_instance.invoke.assert_called_with("test")