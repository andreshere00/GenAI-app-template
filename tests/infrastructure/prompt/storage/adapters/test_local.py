from pathlib import Path
import pytest
from src.infrastructure.prompt.storage.adapters.local import LocalStorageAdapter
from src.domain.prompt.types import PromptTemplate

@pytest.fixture
def temp_prompt_file(tmp_path: Path) -> Path:
    """Fixture to create a temporary prompt file for testing."""
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    file_path = prompt_dir / "test_prompt.txt"
    file_path.write_text("Hello, $name!", encoding="utf-8")
    return file_path

def test_load_template_existing_file_returns_prompt_template(temp_prompt_file: Path):
    """
    Test that load_template correctly reads an existing file and returns a PromptTemplate.
    """
    # Arrange
    base_path = temp_prompt_file.parent
    adapter = LocalStorageAdapter(base_path=base_path)
    relative_path = temp_prompt_file.name

    # Act
    result = adapter.load_template(relative_path)

    # Assert
    assert isinstance(result, PromptTemplate)
    assert result.content == "Hello, $name!"
    assert result.path == str(temp_prompt_file)

def test_load_template_missing_file_raises_file_not_found_error(tmp_path: Path):
    """
    Test that load_template raises FileNotFoundError when the file does not exist.
    """
    # Arrange
    adapter = LocalStorageAdapter(base_path=tmp_path)
    non_existent_path = "ghost.txt"

    # Act & Assert
    with pytest.raises(FileNotFoundError) as exc_info:
        adapter.load_template(non_existent_path)
    
    assert "Prompt file not found at" in str(exc_info.value)

def test_load_template_with_nested_directories(tmp_path: Path):
    """
    Test that load_template handles relative paths within nested directories correctly.
    """
    # Arrange
    nested_dir = tmp_path / "category" / "subcategory"
    nested_dir.mkdir(parents=True)
    file_path = nested_dir / "template.txt"
    file_path.write_text("Nested content", encoding="utf-8")
    
    adapter = LocalStorageAdapter(base_path=tmp_path)
    relative_path = "category/subcategory/template.txt"

    # Act
    result = adapter.load_template(relative_path)

    # Assert
    assert result.content == "Nested content"
    assert Path(result.path) == file_path

def test_init_accepts_both_str_and_path(tmp_path: Path):
    """
    Test that the constructor correctly handles both string and Path objects for base_path.
    """
    # Arrange & Act
    adapter_str = LocalStorageAdapter(base_path=str(tmp_path))
    adapter_path = LocalStorageAdapter(base_path=tmp_path)

    # Assert
    assert adapter_str._base_path == tmp_path
    assert adapter_path._base_path == tmp_path
