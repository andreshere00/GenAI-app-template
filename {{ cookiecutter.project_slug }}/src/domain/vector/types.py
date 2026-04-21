{%- if cookiecutter.vector_db -%}
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class VectorDBProvider(str, Enum):
    """Supported vector database providers."""
{% if "1" in cookiecutter.vector_db %}
    COSMOS_DB = "cosmos-db"
{%- endif %}
{% if "2" in cookiecutter.vector_db %}
    MILVUS = "milvus"
{%- endif %}
{% if "3" in cookiecutter.vector_db %}
    MONGODB = "mongodb"
{%- endif %}
{% if "4" in cookiecutter.vector_db %}
    OPENSEARCH = "opensearch"
{%- endif %}
{% if "5" in cookiecutter.vector_db %}
    PINECONE = "pinecone"
{%- endif %}
{% if "6" in cookiecutter.vector_db %}
    QDRANT = "qdrant"
{%- endif %}
{% if "7" in cookiecutter.vector_db %}
    VERTEX_AI = "vertex-ai"
{%- endif %}


class DistanceMetric(str, Enum):
    """Standardised distance / similarity metrics.

    Adapters translate these into provider-specific values.
    """

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"


@dataclass
class VectorDBConfig:
    """Concrete configuration for vector database connections.

    Not all fields are required for every provider. Check each
    provider's documentation to understand which parameters apply.

    Attributes:
        host: Hostname or IP address of the server.
        port: Port number for the server.
        url: Full URL for cloud-hosted or HTTP-based databases.
        api_key: API key for authentication with cloud services.
        grpc_port: gRPC port for high-performance connections.
        https: Whether to use HTTPS for connections.
        prefix: URL prefix for the database endpoint.
        timeout: Connection timeout in seconds.
        username: Username for basic authentication.
        password: Password for basic authentication.
        database: Database name or project ID.
        collection: Default collection or index name.
        region: Cloud region for hosted services.
        kwargs: Additional provider-specific configuration parameters.
    """

    host: Optional[str] = None
    port: Optional[int] = None
    url: Optional[str] = None
    api_key: Optional[str] = None
    grpc_port: Optional[int] = None
    https: Optional[bool] = None
    prefix: Optional[str] = None
    timeout: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    database: Optional[str] = None
    collection: Optional[str] = None
    region: Optional[str] = None
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class CollectionConfig:
    """Concrete configuration for a vector collection / index.

    Attributes:
        name: Collection or index name (required).
        dimension: Vector dimensionality.
        metric: Distance metric (use DistanceMetric enum values).
        kwargs: Additional provider-specific collection parameters.
    """

    name: str = ""
    dimension: Optional[int] = None
    metric: Optional[str] = None
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorRecord:
    """Concrete vector record for storage operations."""

    id: str = ""
    vector: list[float] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorSearchResult:
    """Concrete vector search hit."""

    id: str = ""
    score: float = 0.0
    payload: dict[str, Any] = field(default_factory=dict)
{%- endif -%}
