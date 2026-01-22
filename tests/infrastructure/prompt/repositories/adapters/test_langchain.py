from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from src.domain.prompt.types import PromptTemplate
from src.infrastructure.prompt.repositories.adapters.langchain import LangchainPromptRepository


@pytest.fixture
def mock_storage_adapter() -> Mock:
    """Fixture to provide a mocked storage adapter."""
    return Mock()


@pytest.fixture
def repository(mock_storage_adapter: Mock) -> LangchainPromptRepository:
    """Fixture to initialize the LangchainPromptRepository with the mock adapter."""
    return LangchainPromptRepository(storage_adapter=mock_storage_adapter)


def test_build_chat_template_valid_input_returns_human_message_by_default(
    repository: LangchainPromptRepository, mock_storage_adapter: Mock
) -> None:
    """
    Test that the repository creates a HumanMessage by default when no role is specified.
    """
    # Arrange
    template_path = "prompts/greeting.txt"
    variables = {"name": "Andres"}
    expected_content = "Hello, Andres!"

    # Mock the raw template return from storage
    mock_storage_adapter.load_template.return_value = PromptTemplate(
        content="Hello, {{name}}!", path=template_path
    )

    # Act
    result = repository.build_chat_template(template_path, variables)

    # Assert
    assert isinstance(result, ChatPromptTemplate)
    messages = result.format_messages()
    assert len(messages) == 1
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == expected_content
    mock_storage_adapter.load_template.assert_called_once_with(template_path)


def test_build_chat_template_system_role_returns_system_message(
    repository: LangchainPromptRepository, mock_storage_adapter: Mock
) -> None:
    """
    Test that providing the 'system' role correctly creates a SystemMessage.
    """
    # Arrange
    template_path = "prompts/system_instruction.txt"
    variables = {"behavior": "helpful"}

    mock_storage_adapter.load_template.return_value = PromptTemplate(
        content="You are a {{behavior}} assistant.", path=template_path
    )

    # Act
    result = repository.build_chat_template(template_path, variables, role="system")

    # Assert
    messages = result.format_messages()
    assert len(messages) == 1
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == "You are a helpful assistant."


def test_build_chat_template_ai_role_returns_ai_message(
    repository: LangchainPromptRepository, mock_storage_adapter: Mock
) -> None:
    """
    Test that providing the 'ai' role correctly creates an AIMessage.
    """
    # Arrange
    template_path = "prompts/example_response.txt"
    variables = {}  # No variables to substitute

    mock_storage_adapter.load_template.return_value = PromptTemplate(
        content="This is a pre-canned response.", path=template_path
    )

    # Act
    result = repository.build_chat_template(template_path, variables, role="ai")

    # Assert
    messages = result.format_messages()
    assert len(messages) == 1
    assert isinstance(messages[0], AIMessage)
    assert messages[0].content == "This is a pre-canned response."


def test_build_chat_template_missing_variable_does_not_crash(
    repository: LangchainPromptRepository, mock_storage_adapter: Mock
) -> None:
    """
    Test that safe_substitute prevents crashing and handles LangChain escaping.
    """
    # Arrange
    template_path = "prompts/complex.txt"
    variables = {"present_var": "Found"}

    mock_storage_adapter.load_template.return_value = PromptTemplate(
        content="Value: {{present_var}}, Missing: {{missing_var}}", path=template_path
    )

    # Act
    result = repository.build_chat_template(template_path, variables)

    # Assert
    messages = result.format_messages()
    assert "Value: Found" in messages[0].content
    # LangChain escapes {{ to { , so we check for the escaped version
    assert "Missing: {missing_var}" in messages[0].content
