from string import Template
from typing import Any

from src.domain.prompt.protocols import PromptStorageAdapter
from src.domain.prompt.types import Prompt


class BasePromptRepository:
    """
    Base repository that handles retrieval and variable substitution.

    This class is framework-agnostic. It returns a pure domain Prompt object.
    """

    def __init__(self, storage_adapter: PromptStorageAdapter) -> None:
        """
        Args:
            storage_adapter: The strategy for retrieving raw templates (S3, Local, etc.).
        """
        self._storage = storage_adapter

    def _build_prompt(self, template_path: str, variables: dict[str, Any]) -> Prompt:
        """
        Internal method to fetch the template and substitute variables.

        This acts as the single source of truth for prompt construction.
        """
        # 1. Fetch raw template via the adapter (Strategy Pattern)
        raw_template = self._storage.load_template(template_path)

        # 2. Perform variable substitution (using standard Python Template)
        processor = Template(raw_template.content)
        final_content = processor.safe_substitute(variables)

        # 3. Return the Domain Entity
        return Prompt(content=final_content)
