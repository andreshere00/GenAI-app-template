"""Domain models for embedding operations."""

from .constants import (
    AZURE_OPENAI_EMBEDDING_PARAM_MAP,
    BEDROCK_EMBEDDING_PARAM_MAP,
    COHERE_EMBEDDING_PARAM_MAP,
    GEMINI_EMBEDDING_PARAM_MAP,
    OPENAI_EMBEDDING_PARAM_MAP,
    VOYAGEAI_EMBEDDING_PARAM_MAP,
    XAI_EMBEDDING_PARAM_MAP,
)
from .protocols import Embedding, EmbeddingConfig
from .types import (
    Embedding as EmbeddingDTO,
    EmbeddingConfig as EmbeddingConfigDTO,
    EmbeddingProvider,
)

__all__: list[str] = [
    # Protocols
    "EmbeddingConfig",
    "Embedding",
    # Concrete types
    "EmbeddingConfigDTO",
    "EmbeddingDTO",
    "EmbeddingProvider",
    # Parameter maps
    "AZURE_OPENAI_EMBEDDING_PARAM_MAP",
    "BEDROCK_EMBEDDING_PARAM_MAP",
    "COHERE_EMBEDDING_PARAM_MAP",
    "GEMINI_EMBEDDING_PARAM_MAP",
    "XAI_EMBEDDING_PARAM_MAP",
    "OPENAI_EMBEDDING_PARAM_MAP",
    "VOYAGEAI_EMBEDDING_PARAM_MAP",
]
