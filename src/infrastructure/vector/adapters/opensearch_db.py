from typing import Any, Optional

from opensearchpy import OpenSearch

from ....domain.vector import (
    CollectionConfig,
    VectorDBConfig,
    VectorDBProvider,
)
from ...utils import resolve_parameters
from ..base import BaseVectorDatabase
from ..factory import VectorDBFactory

OPENSEARCH_ALLOWED_KEYS: set[str] = {
    "hosts",
    "basic_auth",
    "verify_certs",
    "timeout",
    "use_ssl",
}


@VectorDBFactory.register(VectorDBProvider.OPENSEARCH)
class OpenSearchVectorDatabase(BaseVectorDatabase):
    """Wrapper for OpenSearch vector database client.

    Configuration parameters:
        - host: OpenSearch server hostname.
        - port: Server port (default: 9200).
        - https: Enable HTTPS for connections.
        - username / password: Basic authentication credentials.
        - verify_certs: Verify SSL certificates (default: True).
        - timeout: Connection timeout in seconds.
        - url: Full connection URL (alternative to host:port).
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
            allowed_keys={
                "url", "host", "port",
                "https", "username", "password",
            },
        )

        cfg_url = url or config_fields.get("url")
        cfg_host = host or config_fields.get("host")
        cfg_port = port or config_fields.get("port") or 9200
        cfg_https = (
            https
            if https is not None
            else config_fields.get("https")
        )
        cfg_username = username or config_fields.get("username")
        cfg_password = password or config_fields.get("password")

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

        if cfg_username:
            params["basic_auth"] = (cfg_username, cfg_password)

        self.client = OpenSearch(**params)

    # -- Connection --------------------------------------------------

    def connect(self) -> None:
        """Establish connection to OpenSearch.

        Raises:
            ConnectionError: If connection cannot be established.
        """
        try:
            self.client.info()
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to OpenSearch: {e}"
            )

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
            return (
                self.client.cluster.health(timeout="5s")
                is not None
            )
        except Exception:
            return False

    # -- Collection CRUD ---------------------------------------------

    def create_collection(self, config: CollectionConfig) -> None:
        """Create an OpenSearch index with knn vector settings.

        Args:
            config: Collection configuration. Pass additional
                ``mappings`` and ``settings`` via ``config.kwargs``.

        Raises:
            RuntimeError: If creation fails.
        """
        extra: dict[str, Any] = dict(config.kwargs)
        body: dict[str, Any] = extra.pop("body", {})

        if config.dimension is not None and "mappings" not in body:
            space_type = config.metric or "cosinesimil"
            body.setdefault("settings", {}).setdefault(
                "index", {}
            )["knn"] = True
            body["mappings"] = {
                "properties": {
                    "vector": {
                        "type": "knn_vector",
                        "dimension": config.dimension,
                        "method": {
                            "name": "hnsw",
                            "space_type": space_type,
                            "engine": "nmslib",
                        },
                    }
                }
            }

        try:
            self.client.indices.create(
                index=config.name,
                body=body,
                **extra,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create OpenSearch index "
                f"'{config.name}': {exc}"
            ) from exc

    def delete_collection(self, name: str) -> None:
        """Delete an OpenSearch index.

        Args:
            name: Index name.

        Raises:
            RuntimeError: If deletion fails.
        """
        try:
            self.client.indices.delete(index=name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to delete OpenSearch index "
                f"'{name}': {exc}"
            ) from exc

    def list_collections(self) -> list[str]:
        """Return sorted names of all OpenSearch indexes.

        Returns:
            Sorted list of index names.
        """
        aliases: dict[str, Any] = self.client.indices.get_alias(
            index="*"
        )
        return sorted(aliases.keys())

    def has_collection(self, name: str) -> bool:
        """Check whether an OpenSearch index exists.

        Args:
            name: Index name.

        Returns:
            True if the index exists, False otherwise.
        """
        return bool(self.client.indices.exists(index=name))
