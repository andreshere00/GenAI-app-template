{%- if cookiecutter.vector_db -%}
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class VectorDBConfig:
    """Configuration for vector database connections.

    This class defines the standard configuration parameters for initializing
    any vector database provider. Not all fields are required for all
    providers. Check the specific provider's documentation to understand
    which parameters are needed.

    Attributes:
        host: Hostname or IP address of the vector database server.
        port: Port number for the vector database server.
        url: Full URL for cloud-hosted or HTTP-based databases.
        api_key: API key for authentication with cloud services.
        grpc_port: gRPC port for high-performance connections (Qdrant).
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
{%- endif -%}
