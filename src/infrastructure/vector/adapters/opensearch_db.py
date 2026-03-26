from typing import Any, Optional

from opensearchpy import OpenSearch

from ....domain.vector import VectorDBConfig
from ...utils import resolve_parameters
from ..base import BaseVectorDatabase

OPENSEARCH_ALLOWED_KEYS = {
    "hosts",
    "basic_auth",
    "verify_certs",
    "timeout",
    "use_ssl",
}


class OpenSearchVectorDatabase(BaseVectorDatabase):
    """Wrapper for OpenSearch vector database client.

    OpenSearch is an open-source fork of Elasticsearch with native vector
    search capabilities. This adapter manages connections to OpenSearch
    clusters.

    Configuration parameters:
        - host: OpenSearch server hostname (e.g., "localhost")
        - port: OpenSearch server port (default: 9200)
        - https: Enable HTTPS for connections (default: False)
        - username: Username for basic authentication
        - password: Password for basic authentication
        - verify_certs: Verify SSL certificates (default: True)
        - timeout: Connection timeout in seconds
        - url: Full OpenSearch connection URL (alternative to host:port)
    """

    def __init__(
        self,
        config: Optional[VectorDBConfig] = None,
        *,
        host: Optional[str] = None,
        port: Optional[int] = None,
        https: Optional[bool] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        verify_certs: Optional[bool] = None,
        timeout: Optional[int] = None,
        url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize OpenSearch vector database adapter.

        Args:
            config: VectorDBConfig with connection parameters.
            host: OpenSearch server hostname.
            port: OpenSearch server port.
            https: Use HTTPS for connections.
            username: Username for authentication.
            password: Password for authentication.
            verify_certs: Verify SSL certificates.
            timeout: Connection timeout in seconds.
            url: Full OpenSearch connection URL.
            **kwargs: Additional OpenSearch-specific parameters.
        """
        super().__init__(config, **kwargs)

        config_fields = resolve_parameters(
            config,
            allowed_keys={"url", "host", "port", "https", "username", "password"},
        )

        config_url = config_fields.get("url")
        config_host = config_fields.get("host")
        config_port = config_fields.get("port")
        config_https = config_fields.get("https")
        config_username = config_fields.get("username")
        config_password = config_fields.get("password")

        cfg_url = url or config_url
        cfg_host = host or config_host
        cfg_port = port or config_port or 9200
        cfg_https = (
            https
            if https is not None
            else config_https
        )
        cfg_username = username or config_username
        cfg_password = password or config_password

        if cfg_url:
            resolved_hosts: list[Any] = [cfg_url]
        elif cfg_host:
            resolved_hosts = [
                {
                    "host": cfg_host,
                    "port": cfg_port,
                    "scheme": "https" if cfg_https else "http",
                }
            ]
        else:
            resolved_hosts = []

        params = resolve_parameters(
            config,
            allowed_keys=OPENSEARCH_ALLOWED_KEYS,
            hosts=resolved_hosts,
            use_ssl=cfg_https,
            verify_certs=verify_certs,
            timeout=timeout,
            **kwargs,
        )

        # Handle authentication
        if cfg_username:
            params["basic_auth"] = (cfg_username, cfg_password)

        self.client = OpenSearch(**params)

    def connect(self) -> None:
        """Establish connection to OpenSearch.

        The OpenSearch client constructor establishes the connection.
        This method validates the connection.
        """
        try:
            self.client.info()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to OpenSearch: {e}")

    def disconnect(self) -> None:
        """Close the OpenSearch connection."""
        if self.client:
            self.client.close()

    def health(self) -> bool:
        """Check if OpenSearch connection is healthy.

        Returns:
            True if database is accessible, False otherwise.
        """
        try:
            return self.client.cluster.health(timeout="5s") is not None
        except Exception:
            return False
