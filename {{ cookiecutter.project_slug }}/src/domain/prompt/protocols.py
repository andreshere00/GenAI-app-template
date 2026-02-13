from typing import Any, Protocol

from src.domain.prompt.types import Prompt, PromptTemplate


class PromptStorageAdapter(Protocol):
    """Interface for storage backends retrieving raw prompt templates."""

    def load_template(self, path: str) -> PromptTemplate:
        """
        Loads a raw prompt template from the specific storage.

        Args:
            path: The identifier or path to the prompt file.

        Returns:
            PromptTemplate: The raw template object.

        Raises:
            FileNotFoundError: If the template cannot be found.
        """
        ...


class PromptRepository(Protocol):
    """Interface for managing prompt retrieval and construction."""

    def get_prompt(self, template_path: str, variables: dict[str, Any]) -> Prompt:
        """
        Constructs a final prompt by fetching a template and applying variables.

        Args:
            template_path: The path identifier for the template.
            variables: A dictionary of variables to inject into the template.

        Returns:
            Prompt: The constructed prompt with variables replaced.
        """
        ...
