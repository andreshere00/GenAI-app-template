"""Domain models for embedding operations."""

from .protocols import Embedding, EmbeddingConfig
from .types import (
    Embedding as EmbeddingDTO,
    EmbeddingConfig as EmbeddingConfigDTO,
)
{% if cookiecutter.embedding_providers %}
from .types import EmbeddingProvider
from .constants import (
    {% if "1" in cookiecutter.embedding_providers %}
    AZURE_OPENAI_EMBEDDING_PARAM_MAP,
    {%- endif -%}
    {% if "2" in cookiecutter.embedding_providers %}
    BEDROCK_EMBEDDING_PARAM_MAP,
    {%- endif -%}
    {% if "3" in cookiecutter.embedding_providers %}
    COHERE_EMBEDDING_PARAM_MAP,
    {%- endif -%}
    {% if "4" in cookiecutter.embedding_providers %}
    GEMINI_EMBEDDING_PARAM_MAP,
    {%- endif -%}
    {% if "5" in cookiecutter.embedding_providers %}
    XAI_EMBEDDING_PARAM_MAP,
    {%- endif -%}
    {% if "7" in cookiecutter.embedding_providers %}
    OPENAI_EMBEDDING_PARAM_MAP,
    {%- endif -%}
    {% if "8" in cookiecutter.embedding_providers %}
    VOYAGEAI_EMBEDDING_PARAM_MAP,
    {% endif %}
)
{% endif %}

__all__: list[str] = [
    # Protocols
    "EmbeddingConfig",
    "Embedding",
    # Concrete types
    "EmbeddingConfigDTO",
    "EmbeddingDTO",
    {% if cookiecutter.embedding_providers %}
    "EmbeddingProvider",
    {%- endif -%}
    {% if "1" in cookiecutter.embedding_providers %}
    "AZURE_OPENAI_EMBEDDING_PARAM_MAP",
    {%- endif -%}
    {% if "2" in cookiecutter.embedding_providers %}
    "BEDROCK_EMBEDDING_PARAM_MAP",
    {%- endif -%}
    {% if "3" in cookiecutter.embedding_providers %}
    "COHERE_EMBEDDING_PARAM_MAP",
    {%- endif -%}
    {% if "4" in cookiecutter.embedding_providers %}
    "GEMINI_EMBEDDING_PARAM_MAP",
    {%- endif -%}
    {% if "5" in cookiecutter.embedding_providers %}
    "XAI_EMBEDDING_PARAM_MAP",
    {%- endif -%}
    {% if "7" in cookiecutter.embedding_providers %}
    "OPENAI_EMBEDDING_PARAM_MAP",
    {%- endif -%}
    {% if "8" in cookiecutter.embedding_providers %}
    "VOYAGEAI_EMBEDDING_PARAM_MAP",
    {%- endif -%}
]
