from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

from pydantic import SecretStr

{% if cookiecutter.embedding_providers %}
class EmbeddingProvider(str, Enum):
    """Supported embedding model providers."""
    {% if "1" in cookiecutter.embedding_providers %}
    AZURE = "azure-openai"
    {%- endif -%}
    {% if "2" in cookiecutter.embedding_providers %}
    AWS = "bedrock"
    {%- endif -%}
    {% if "3" in cookiecutter.embedding_providers %}
    COHERE = "cohere"
    {%- endif -%}
    {% if "4" in cookiecutter.embedding_providers %}
    GOOGLE = "gemini"
    {%- endif -%}
    {% if "5" in cookiecutter.embedding_providers %}
    XAI = "grok"
    {%- endif -%}
    {% if "6" in cookiecutter.embedding_providers %}
    HUGGINGFACE = "ollama"
    {%- endif -%}
    {% if "7" in cookiecutter.embedding_providers %}
    OPENAI = "openai"
    {%- endif -%}
    {% if "8" in cookiecutter.embedding_providers %}
    VOYAGEAI = "voyageai"
    {%- endif -%}
{%- endif -%}


@dataclass
class EmbeddingConfig:
    """Concrete configuration for embedding model connections.

    Attributes:
        api_key: API key for authentication.
        model: The embedding model name or deployment.
        base_url: Base URL for the API endpoint.
        organization: Organization identifier.
        azure_endpoint: Azure-specific endpoint URL.
        azure_deployment: Azure deployment name.
        api_version: API version string.
        timeout: Request timeout in seconds.
        max_retries: Maximum number of retry attempts.
        model_kwargs: Additional provider-specific parameters.
    """

    api_key: Optional[Union[str, SecretStr]] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    organization: Optional[str] = None
    azure_endpoint: Optional[str] = None
    azure_deployment: Optional[str] = None
    api_version: Optional[str] = None
    timeout: Optional[Union[float, int]] = None
    max_retries: Optional[int] = None
    model_kwargs: Optional[dict[str, Any]] = None


@dataclass
class Embedding:
    """Concrete embedding output extending SplitterMR SplitterOutput.

    Attributes:
        embeddings: Embedding vectors, one list of floats per chunk.
        embedding_id: Unique identifier for this embedding batch.
        chunks: Text chunks from the splitter.
        chunk_id: Unique IDs corresponding to each chunk.
        document_name: Name of the source document.
        document_path: Path to the source document.
        document_id: Unique identifier for the document.
        conversion_method: Method used for document conversion.
        reader_method: Method used for reading the document.
        ocr_method: OCR method used, if any.
        split_method: Method used to split the document.
        split_params: Parameters used during splitting.
        metadata: Additional metadata.
    """

    embeddings: list[list[float]] = field(default_factory=list)
    embedding_id: str = ""
    chunks: list[str] = field(default_factory=list)
    chunk_id: list[str] = field(default_factory=list)
    document_name: Optional[str] = None
    document_path: str = ""
    document_id: Optional[str] = None
    conversion_method: Optional[str] = None
    reader_method: Optional[str] = None
    ocr_method: Optional[str] = None
    split_method: str = ""
    split_params: Optional[dict[str, Any]] = None
    metadata: Optional[dict[str, Any]] = None
