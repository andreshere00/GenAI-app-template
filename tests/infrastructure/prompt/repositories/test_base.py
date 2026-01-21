from unittest.mock import Mock

import pytest

from src.domain.prompt.types import Prompt, PromptTemplate
from src.infrastructure.prompt.repositories.base import BasePromptRepository


class ConcretePromptRepository(BasePromptRepository):
    """
    Minimal concrete implementation to expose the protected _build_prompt method
    for testing purposes.
    """

    def build(self, template_path: str, variables: dict) -> Prompt:
        return self._build_prompt(template_path, variables)


@pytest.fixture
def mock_storage_adapter() -> Mock:
    """Fixture to provide a mocked storage adapter."""
    return Mock()


@pytest.fixture
def repository(mock_storage_adapter: Mock) -> ConcretePromptRepository:
    """Fixture to initialize the concrete repository with the mock adapter."""
    return ConcretePromptRepository(storage_adapter=mock_storage_adapter)


def test_build_prompt_valid_variables_returns_substituted_content(
    repository: ConcretePromptRepository, mock_storage_adapter: Mock
) -> None:
    """
    Test that the repository correctly fetches a template and substitutes variables.
    """
    # Arrange
    template_path = "prompts/welcome.txt"
    variables = {"user": "Alice", "role": "Admin"}
    raw_content = "Welcome $user, your role is $role."
    expected_content = "Welcome Alice, your role is Admin."

    mock_storage_adapter.load_template.return_value = PromptTemplate(
        content=raw_content, path=template_path
    )

    # Act
    result = repository.build(template_path, variables)

    # Assert
    assert isinstance(result, Prompt)
    assert result.content == expected_content
    mock_storage_adapter.load_template.assert_called_once_with(template_path)


def test_build_prompt_missing_variables_performs_safe_substitution(
    repository: ConcretePromptRepository, mock_storage_adapter: Mock
) -> None:
    """
    Test that missing variables do not crash the application and are left as placeholders.
    """
    # Arrange
    template_path = "prompts/email.txt"
    variables = {"name": "Bob"}  # 'date' is missing
    raw_content = "Hello $name, today is $date."
    expected_content = "Hello Bob, today is $date."

    mock_storage_adapter.load_template.return_value = PromptTemplate(
        content=raw_content, path=template_path
    )

    # Act
    result = repository.build(template_path, variables)

    # Assert
    assert result.content == expected_content


def test_build_prompt_extra_variables_ignores_them(
    repository: ConcretePromptRepository, mock_storage_adapter: Mock
) -> None:
    """
    Test that providing extra variables not present in the template does not cause errors.
    """
    # Arrange
    template_path = "prompts/simple.txt"
    variables = {"name": "Charlie", "unused_var": "I am ignored"}
    raw_content = "Hi $name."
    expected_content = "Hi Charlie."

    mock_storage_adapter.load_template.return_value = PromptTemplate(
        content=raw_content, path=template_path
    )

    # Act
    result = repository.build(template_path, variables)

    # Assert
    assert result.content == expected_content


def test_build_prompt_storage_error_propagates_exception(
    repository: ConcretePromptRepository, mock_storage_adapter: Mock
) -> None:
    """
    Test that exceptions raised by the storage adapter (e.g., FileNotFoundError)
    bubble up correctly.
    """
    # Arrange
    template_path = "prompts/non_existent.txt"
    mock_storage_adapter.load_template.side_effect = FileNotFoundError("File not found")

    # Act & Assert
    with pytest.raises(FileNotFoundError):
        repository.build(template_path, {})