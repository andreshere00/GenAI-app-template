from typing import Any, Optional

from qdrant_client import QdrantClient

from ....domain.vector import VectorDBConfig
from ...utils import resolve_parameters
from ..base import BaseVectorDatabase

QDRANT_ALLOWED_KEYS = {
    "url",
    "host",
    "port",
    "grpc_port",
    "prefer_grpc",
    "https",
    "api_key",
    "prefix",
    "timeout",
    "path",
}


class QdrantVectorDatabase(BaseVectorDatabase):
    """Wrapper for Qdrant vector database client.

    Qdrant is an open-source vector database optimized for similarity search
    at scale. This adapter provides a unified interface for initializing and
    managing Qdrant connections.

    Configuration parameters:
        - url: HTTP endpoint URL (e.g., "http://localhost:6333")
        - api_key: API key for cloud deployments
        - host: Host address for local/self-hosted instances
        - port: Port number (default: 6333)
        - grpc_port: gRPC port for high-performance connections
        - prefer_grpc: Use gRPC instead of REST (default: False)
        - https: Use HTTPS for connections
        - prefix: URL path prefix
        - timeout: Request timeout in seconds
        - path: Path for in-memory/persisted local mode
    """

    def __init__(
        self,
        config: Optional[VectorDBConfig] = None,
        *,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        grpc_port: Optional[int] = None,
        prefer_grpc: Optional[bool] = None,
        https: Optional[bool] = None,
        prefix: Optional[str] = None,
        timeout: Optional[int] = None,
        path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Qdrant vector database adapter.

        Args:
            config: VectorDBConfig with connection parameters.
            url: HTTP endpoint URL for Qdrant server.
            api_key: API key for Qdrant Cloud.
            host: Hostname or IP address of Qdrant server.
            port: Port number for HTTP API.
            grpc_port: Port number for gRPC API.
            prefer_grpc: Prefer gRPC over REST API.
            https: Use HTTPS for connections.
            prefix: URL path prefix for Qdrant API.
            timeout: Request timeout in seconds.
            path: Local path for in-memory or persisted mode.
            **kwargs: Additional Qdrant-specific parameters.
        """
        super().__init__(config, **kwargs)

        params = resolve_parameters(
            config,
            allowed_keys=QDRANT_ALLOWED_KEYS,
            url=url,
            api_key=api_key,
            host=host,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            https=https,
            prefix=prefix,
            timeout=timeout,
            path=path,
            **kwargs,
        )

        self.client = QdrantClient(**params)

    def connect(self) -> None:
        """Establish connection to Qdrant.

        The QdrantClient constructor establishes the connection.
        This method validates the connection.
        """
        try:
            self.client.get_collections()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Qdrant: {e}")

    def disconnect(self) -> None:
        """Close the Qdrant connection."""
        if self.client:
            self.client.close()

    def health(self) -> bool:
        """Check if Qdrant connection is healthy.

        Returns:
            True if database is accessible, False otherwise.
        """
        try:
            return self.client.http_client is not None
        except Exception:
            return False
