from enum import Enum


class LLMProvider(str, Enum):
    OPENAI = "openai"
    AZURE = "azure-openai"
    ANTHROPIC = "claude"
    GOOGLE = "gemini"
    AWS = "bedrock"
    XAI = "grok"
    HUGGINGFACE = "ollama"
