from pydantic import BaseModel, ConfigDict


class PromptTemplate(BaseModel):
    """Represents a raw template retrieved from storage."""

    model_config = ConfigDict(frozen=True)

    content: str
    path: str


class Prompt(BaseModel):
    """Represents the final processed prompt ready for the LLM."""

    model_config = ConfigDict(frozen=True)

    content: str
