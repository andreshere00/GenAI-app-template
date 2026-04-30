from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from src.domain.embedding.protocols import EmbeddingConfig
from src.infrastructure.embedding.factory import EmbeddingFactory

# ---- Mocks, fixtures & helpers ---- #


@pytest.fixture(autouse=True)
def clean_registry():
    """Backup and restore the factory registry between tests."""
    original_registry = EmbeddingFactory._registry.copy()
    yield
    EmbeddingFactory._registry = original_registry


class MockModel:
    """Simple mock model to verify instantiation params."""

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        **kwargs: Any,
    ):
        self.config = config
        self.kwargs = kwargs


# ---- Happy path ---- #


class TestEmbeddingFactory:
    def test_register_decorator_adds_to_registry(self):
        """Test that register decorator correctly adds a class."""
        provider_name = "test_provider"

        @EmbeddingFactory.register(provider_name)
        class TestModel:
            pass

        assert provider_name in EmbeddingFactory._registry
        assert EmbeddingFactory._registry[provider_name] == TestModel

    def test_create_registered_provider_returns_instance(self):
        """Test successful creation of a registered model."""
        provider_name = "mock_embed"
        EmbeddingFactory.register(provider_name)(MockModel)

        instance = EmbeddingFactory.create(
            provider_name, model="text-embedding-3-small"
        )

        assert isinstance(instance, MockModel)
        assert instance.kwargs["model"] == "text-embedding-3-small"

    def test_create_with_config_passes_config(self):
        """Test creation passing a configuration object."""
        provider_name = "config_embed"
        config_mock = MagicMock(spec=EmbeddingConfig)
        EmbeddingFactory.register(provider_name)(MockModel)

        instance = EmbeddingFactory.create(
            provider_name, config=config_mock
        )

        assert instance.config == config_mock

    def test_create_forwards_all_kwargs(self):
        """Test that all kwargs are forwarded to the constructor."""
        provider_name = "kwargs_embed"
        EmbeddingFactory.register(provider_name)(MockModel)

        instance = EmbeddingFactory.create(
            provider_name, arg1="v1", arg2=42, nested={"k": 1}
        )

        assert instance.kwargs["arg1"] == "v1"
        assert instance.kwargs["arg2"] == 42
        assert instance.kwargs["nested"] == {"k": 1}

    # ---- Error paths ---- #

    def test_create_unregistered_provider_raises_error(self):
        """Test that an unregistered provider raises ValueError."""
        with pytest.raises(ValueError, match="is not registered"):
            EmbeddingFactory.create("non_existent_provider")

    # ---- Edge cases ---- #

    def test_register_same_key_twice_overwrites(self):
        """Test that re-registering overwrites the previous class."""
        provider_name = "dup_embed"

        class ModelA:
            pass

        class ModelB:
            pass

        EmbeddingFactory.register(provider_name)(ModelA)
        assert EmbeddingFactory._registry[provider_name] == ModelA

        EmbeddingFactory.register(provider_name)(ModelB)
        assert EmbeddingFactory._registry[provider_name] == ModelB
