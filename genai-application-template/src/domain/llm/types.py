from enum import Enum


class LLMProvider(str, Enum):
    
    ANTHROPIC = "claude"
    AZURE = "azure-openai"
    XAI = "grok"