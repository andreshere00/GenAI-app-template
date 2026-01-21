from pathlib import Path

from .....domain.prompt.types import PromptTemplate
from ..base import BaseStorageAdapter


class LocalStorageAdapter(BaseStorageAdapter):
    """Implementation of storage adapter for the local file system."""

    def __init__(self, base_path: str | Path) -> None:
        """
        Initializes the local storage adapter.

        Args:
            base_path: The root directory where prompts are stored.
        """
        self._base_path = Path(base_path)

    def load_template(self, path: str) -> PromptTemplate:
        """
        Reads a file from the local filesystem.

        Args:
            path: The relative path to the file from base_path.

        Returns:
            PromptTemplate: The content of the file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        full_path = self._base_path / path

        if not full_path.exists():
            raise FileNotFoundError(f"Prompt file not found at: {full_path}")

        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()

        return PromptTemplate(content=content, path=str(full_path))
