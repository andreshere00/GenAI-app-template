import pytest

from src.domain.vector import VectorDBConfig
from src.infrastructure.vector.base import BaseVectorDatabase


class ConcreteVectorDatabase(BaseVectorDatabase):
    """Test double for validating the abstract base class contract."""

    def connect(self) -> None:
        self.client = object()

    def disconnect(self) -> None:
        self.client = None

    def health(self) -> bool:
        return self.client is not None


class TestBaseVectorDatabase:
    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            BaseVectorDatabase()

    def test_concrete_subclass_initializes_and_manages_client_state(self):
        config = VectorDBConfig(host="localhost", port=6333)
        adapter = ConcreteVectorDatabase(config=config)

        assert adapter.config == config
        assert adapter.client is None

        adapter.connect()
        assert adapter.health() is True

        adapter.disconnect()
        assert adapter.health() is False
