from unittest.mock import patch

import pytest

from src.infrastructure.llm.adapters.bedrock import BedrockModel

# --- Mocks & Fixtures ---


@pytest.fixture
def mock_chat_bedrock():
    """Patches the ChatBedrock class to prevent real API calls.

    Returns:
        MagicMock: The mock object for the ChatBedrock class.
    """
    with patch("src.infrastructure.llm.adapters.bedrock.ChatBedrock") as mock:
        yield mock


# --- Test Suite ---


class TestBedrockModel:
    def test_init_valid_config_sets_client_params(
        self, config_factory, mock_chat_bedrock
    ):
        """Test instantiation using only the configuration object."""
        config = config_factory(
            model="anthropic.claude-v2",
            temperature=0.7,
            api_key="ignored-key",  # Should be removed
        )

        BedrockModel(config=config)

        mock_chat_bedrock.assert_called_once()
        call_kwargs = mock_chat_bedrock.call_args.kwargs

        # Verify mapping: 'model' -> 'model_id' (Standard Bedrock param)
        assert call_kwargs["model_id"] == "anthropic.claude-v2"
        # Verify 'api_key' is explicitly removed for Bedrock
        assert "api_key" not in call_kwargs
        # Verify other params are passed
        assert call_kwargs["temperature"] == 0.7

    def test_init_valid_kwargs_sets_client_params(self, mock_chat_bedrock):
        """Test instantiation using only explicit keyword arguments."""
        BedrockModel(model="anthropic.claude-3-sonnet", temperature=0.5)

        mock_chat_bedrock.assert_called_once()
        call_kwargs = mock_chat_bedrock.call_args.kwargs

        assert call_kwargs["model_id"] == "anthropic.claude-3-sonnet"
        assert call_kwargs["temperature"] == 0.5

    def test_init_config_and_kwargs_prioritizes_kwargs(
        self, config_factory, mock_chat_bedrock
    ):
        """Test that explicit kwargs override configuration values."""
        config = config_factory(model="anthropic.claude-v2", temperature=0.1)

        # Override model and temperature
        BedrockModel(config=config, model="anthropic.claude-3-haiku", temperature=0.9)

        call_kwargs = mock_chat_bedrock.call_args.kwargs

        assert call_kwargs["model_id"] == "anthropic.claude-3-haiku"  # Overridden
        assert call_kwargs["temperature"] == 0.9  # Overridden

    def test_init_remappable_keys_maps_to_langchain_params(self, mock_chat_bedrock):
        """Test that generic keys are correctly mapped to LangChain Bedrock keys."""
        # 'model' is mapped to 'model_id' in BEDROCK_PARAM_MAP
        BedrockModel(model="meta.llama2-70b-chat-v1")

        call_kwargs = mock_chat_bedrock.call_args.kwargs

        assert "model" not in call_kwargs
        assert call_kwargs["model_id"] == "meta.llama2-70b-chat-v1"

    def test_init_removes_api_key_param(self, mock_chat_bedrock):
        """Test that the api_key parameter is strictly removed before client init."""
        # Bedrock uses boto3 credentials, passing api_key often causes errors
        BedrockModel(api_key="some-secret-key", model="amazon.titan-text-express-v1")

        call_kwargs = mock_chat_bedrock.call_args.kwargs

        assert "api_key" not in call_kwargs
        assert "aws_api_key" not in call_kwargs

    def test_getattr_valid_method_delegates_to_client(self, mock_chat_bedrock):
        """Test that the underlying client methods are accessible via the wrapper."""
        mock_client_instance = mock_chat_bedrock.return_value
        model = BedrockModel(model="anthropic.claude-v2")

        # 1. Access the client attribute directly
        assert model.client == mock_client_instance

        # 2. Simulate calling a standard LangChain method
        dummy_input = "Hello Bedrock"
        model.client.invoke(dummy_input)

        mock_client_instance.invoke.assert_called_once_with(dummy_input)

    def test_register_class_loading_registers_in_factory(self):
        """Test that the class is correctly registered in the factory."""
        from src.domain.llm.types import LLMProvider
        from src.infrastructure.llm.factory import LlmFactory

        assert LLMProvider.AWS in LlmFactory._registry
        assert LlmFactory._registry[LLMProvider.AWS] == BedrockModel

    def test_client_runnable_methods_are_accessible(self, mock_chat_bedrock):
        """Test that the model client implements standard LangChain Runnable methods."""
        mock_instance = mock_chat_bedrock.return_value
        model = BedrockModel(model="anthropic.claude-v2")

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
