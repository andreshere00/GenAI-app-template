from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from src.domain.llm.protocols import ModelConfig
from src.infrastructure.llm.factory import LlmFactory

# --- Fixtures ---


@pytest.fixture(autouse=True)
def clean_registry():
    """Fixture to backup and restore the factory registry.

    This ensures that tests do not pollute the global registry state,
    preventing side effects between tests.
    """
    # 1. Backup original registry
    original_registry = LlmFactory._registry.copy()

    yield

    # 2. Restore registry after test
    LlmFactory._registry = original_registry


class MockModel:
    """A simple mock model class to verify instantiation params."""

    def __init__(self, config: Optional[ModelConfig] = None, **kwargs: Any):
        self.config = config
        self.kwargs = kwargs


# --- Test Suite ---


class TestLlmFactory:

    def test_register_decorator(self):
        """Test that the register decorator correctly adds a class to the registry."""
        provider_name = "test_provider"

        # Apply decorator manually
        @LlmFactory.register(provider_name)
        class TestProviderModel:
            pass

        assert provider_name in LlmFactory._registry
        assert LlmFactory._registry[provider_name] == TestProviderModel

    def test_create_success(self):
        """Test successful creation of a registered model."""
        provider_name = "mock_provider"

        # Register the mock model
        LlmFactory.register(provider_name)(MockModel)

        # Create instance
        instance = LlmFactory.create(provider_name, temperature=0.5)

        assert isinstance(instance, MockModel)
        assert instance.kwargs["temperature"] == 0.5

    def test_create_with_config(self):
        """Test creation passing a configuration object."""
        provider_name = "config_provider"
        config_mock = MagicMock(spec=ModelConfig)

        LlmFactory.register(provider_name)(MockModel)

        instance = LlmFactory.create(provider_name, config=config_mock)

        assert instance.config == config_mock

    def test_create_unregistered_provider_raises_error(self):
        """Test that requesting an unregistered provider raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            LlmFactory.create("non_existent_provider")

        assert "is not registered" in str(exc_info.value)

    def test_create_passes_all_kwargs(self):
        """Test that all kwargs are forwarded to the model constructor."""
        provider_name = "kwargs_provider"
        LlmFactory.register(provider_name)(MockModel)

        instance = LlmFactory.create(provider_name, arg1="value1", arg2=123, nested={"a": 1})

        assert instance.kwargs["arg1"] == "value1"
        assert instance.kwargs["arg2"] == 123
        assert instance.kwargs["nested"] == {"a": 1}

    def test_registry_overwrite(self):
        """Test that registering the same key twice overwrites the previous class.

        (This behavior depends on desired design; standard dict behavior overwrites)
        """
        provider_name = "duplicate_provider"

        class ModelA:
            pass

        class ModelB:
            pass

        LlmFactory.register(provider_name)(ModelA)
        assert LlmFactory._registry[provider_name] == ModelA

        # Overwrite
        LlmFactory.register(provider_name)(ModelB)
        assert LlmFactory._registry[provider_name] == ModelB
