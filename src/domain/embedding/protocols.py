from __future__ import annotations

from typing import Any, Optional, Protocol, Union

from pydantic import SecretStr


class EmbeddingConfig(Protocol):
    """Protocol defining the configuration shape for embedding models.

    This protocol is extensible. Generic fields like 'model' or 'base_url'
    can be mapped to provider-specific arguments (e.g., 'azure_deployment')
    via the adapter classes.
    """

    api_key: Union[str, SecretStr]
    model: Optional[str]

    # Standard connection
    base_url: Optional[str]
    organization: Optional[str]

    # Azure specifics
    azure_endpoint: Optional[str]
    azure_deployment: Optional[str]
    api_version: Optional[str]

    # Common configuration
    timeout: Optional[Union[float, int]]
    max_retries: Optional[int]
    model_kwargs: Optional[dict[str, Any]]


class Embedding(Protocol):
    """Protocol defining the output structure of an embedding operation.

    Extends the SplitterMR ``SplitterOutput`` structure with embedding
    vectors and a unique embedding identifier.
    """

    embeddings: list[list[float]]
    embedding_id: str
    chunks: list[str]
    chunk_id: list[str]
    document_name: Optional[str]
    document_path: str
    document_id: Optional[str]
    conversion_method: Optional[str]
    reader_method: Optional[str]
    ocr_method: Optional[str]
    split_method: str
    split_params: Optional[dict[str, Any]]
    metadata: Optional[dict[str, Any]]
