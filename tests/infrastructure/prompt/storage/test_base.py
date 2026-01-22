import pytest
from src.infrastructure.prompt.storage.base import BaseStorageAdapter
from src.domain.prompt.types import PromptTemplate


class ConcreteStorageAdapter(BaseStorageAdapter):
    """
    Concrete implementation of BaseStorageAdapter for testing purposes.
    Implements the abstract methods required by the PromptStorageAdapter protocol.
    """

    def load_template(self, path: str) -> PromptTemplate:
        return PromptTemplate(content="test content", path=path)


def test_base_storage_adapter_is_abstract():
    """
    Test that BaseStorageAdapter cannot be instantiated directly because it is an ABC.
    """
    # Act & Assert
    with pytest.raises(TypeError):
        # This should fail because it doesn't implement load_template
        # required by the protocol/ABC structure
        BaseStorageAdapter()


def test_concrete_adapter_inherits_from_base():
    """
    Test that a concrete implementation correctly identifies as an instance
    of BaseStorageAdapter.
    """
    # Arrange
    adapter = ConcreteStorageAdapter()

    # Assert
    assert isinstance(adapter, BaseStorageAdapter)
    assert isinstance(adapter, ConcreteStorageAdapter)


def test_concrete_adapter_implements_protocol_method():
    """
    Test that the concrete adapter can execute the methods defined in the protocol.
    """
    # Arrange
    adapter = ConcreteStorageAdapter()
    test_path = "prompts/test.txt"

    # Act
    result = adapter.load_template(test_path)

    # Assert
    assert result.content == "test content"
    assert result.path == test_path
