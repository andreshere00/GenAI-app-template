from typing import Any, Optional

from pymilvus import MilvusClient

from ....domain.vector import VectorDBConfig
from ...utils import resolve_parameters
from ..base import BaseVectorDatabase

MILVUS_ALLOWED_KEYS = {
    "uri",
    "token",
    "db_name",
    "timeout",
}


class MilvusVectorDatabase(BaseVectorDatabase):
    """Wrapper for Milvus vector database client.

    Milvus is an open-source vector database designed for similarity search at
    scale. This adapter provides a unified interface for initializing and
    managing Milvus connections.

    Configuration parameters:
        - host: Hostname of Milvus server (default: "localhost")
        - port: Port number (default: 19530)
        - database: Database name (default: "default")
        - uri: Connection URI string (alternative to host:port)
        - username: Username for authentication (optional)
        - password: Password for authentication (optional)
        - timeout: Connection timeout in seconds (default: 60)
    """

    def __init__(
        self,
        config: Optional[VectorDBConfig] = None,
        *,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Milvus vector database adapter.

        Args:
            config: VectorDBConfig with connection parameters.
            host: Milvus server hostname.
            port: Milvus server port.
            database: Database name to connect to.
            uri: Alternative connection URI string.
            username: Username for authentication.
            password: Password for authentication.
            timeout: Connection timeout in seconds.
            **kwargs: Additional Milvus-specific parameters.
        """
        super().__init__(config, **kwargs)

        config_fields = resolve_parameters(
            config,
            allowed_keys={"host", "port", "username", "password"},
        )
        resolved_uri = uri
        config_host = config_fields.get("host")
        config_port = config_fields.get("port")
        config_user = config_fields.get("username")
        config_password = config_fields.get("password")
        if resolved_uri is None:
            cfg_host = host or config_host
            cfg_port = port or config_port or 19530
            if cfg_host:
                resolved_uri = f"http://{cfg_host}:{cfg_port}"

        resolved_token: Optional[str] = None
        resolved_username = username or config_user
        resolved_password = password or config_password
        if resolved_username and resolved_password:
            resolved_token = f"{resolved_username}:{resolved_password}"

        params = resolve_parameters(
            config,
            allowed_keys=MILVUS_ALLOWED_KEYS,
            aliases={"database": "db_name"},
            uri=resolved_uri,
            db_name=database,
            token=resolved_token,
            timeout=timeout,
            **kwargs,
        )

        self.client = MilvusClient(**params)

    def connect(self) -> None:
        """Establish connection to Milvus database.

        The MilvusClient constructor already establishes the connection.
        """

    def disconnect(self) -> None:
        """Close the Milvus connection."""
        if self.client:
            self.client.close()

    def health(self) -> bool:
        """Check if Milvus connection is healthy.

        Returns:
            True if database is accessible, False otherwise.
        """
        try:
            return self.client.get_server_version() is not None
        except Exception:
            return False
