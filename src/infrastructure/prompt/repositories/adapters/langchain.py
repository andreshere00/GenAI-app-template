from typing import Any, Literal

from langchain_core.prompts import ChatPromptTemplate

from src.infrastructure.prompt.repositories.base import BasePromptRepository
from src.domain.prompt.types import Prompt


class LangchainPromptRepository(BasePromptRepository):
    """
    LangChain-specific repository.

    It extends the base logic to return objects ready for LCEL chains.
    """

    def build_chat_template(
        self,
        template_path: str,
        variables: dict[str, Any],
        role: Literal["system", "human", "ai"] = "human",
    ) -> ChatPromptTemplate:
        """
        Retrieves a prompt and converts it into a LangChain ChatPromptTemplate.

        Args:
            template_path: Path to the template file.
            variables: Dictionary of input variables.
            role: The role to assign to the message (default: human).

        Returns:
            ChatPromptTemplate: Ready to be used in LCEL (e.g., prompt | llm).
        """
        # 1. Reuse the base logic to get the processed domain entity
        domain_prompt: Prompt = self._build_prompt(template_path, variables)

        # 2. Convert to LangChain ChatPromptTemplate
        return ChatPromptTemplate.from_messages(
            [(role, domain_prompt.content)],
            template_format="mustache",
        )
