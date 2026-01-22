import re
from typing import Any

from src.domain.prompt.protocols import PromptStorageAdapter
from src.domain.prompt.types import Prompt


class BasePromptRepository:
    """Base repository using regex for extensible variable substitution."""

    def __init__(
        self,
        storage_adapter: PromptStorageAdapter,
        variable_pattern: str = r"\{\{\s*(\w+)\s*\}\}",
    ) -> None:
        """
        Initialize the BasePromptRepository class.

        Args:
            storage_adapter: Strategy for retrieving raw templates.
            variable_pattern: Regex pattern to identify variables (defaults to double brackets).
        """
        self._storage = storage_adapter
        self._pattern = re.compile(variable_pattern)

    def _build_prompt(self, template_path: str, variables: dict[str, Any]) -> Prompt:
        """
        Fetches template and replaces variables based on the configured pattern.

        Args:
            template_path: Identifier for the raw template.
            variables: Data to inject into the template.

        Returns:
            Prompt: Processed content as a Domain Entity.
        """
        raw_template = self._storage.load_template(template_path)
        content = raw_template.content

        # Substitution using the compiled regex pattern
        final_content = self._pattern.sub(
            lambda match: str(variables.get(match.group(1), match.group(0))), content
        )

        return Prompt(content=final_content)
