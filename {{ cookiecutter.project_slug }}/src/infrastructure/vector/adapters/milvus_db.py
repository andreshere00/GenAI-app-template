{%- if "2" in cookiecutter.vector_db -%}
from typing import Any, Optional

from pymilvus import MilvusClient

from ....domain.vector import (
    MILVUS_PARAM_MAP,
    CollectionConfig,
    DistanceMetric,
    VectorRecord,
    VectorSearchResultDTO as VectorSearchResult,
    VectorDBConfig,
    VectorDBProvider,
)
from ...utils import resolve_parameters
from ..base import BaseVectorDatabase
from ..factory import VectorDBFactory

MILVUS_ALLOWED_KEYS: set[str] = {
    "uri",
    "token",
    "db_name",
    "timeout",
}

_METRIC_MAP: dict[DistanceMetric, str] = {
    DistanceMetric.COSINE: "COSINE",
    DistanceMetric.EUCLIDEAN: "L2",
    DistanceMetric.DOT_PRODUCT: "IP",
}


@VectorDBFactory.register(VectorDBProvider.MILVUS)
class MilvusVectorDatabase(BaseVectorDatabase):
    """Wrapper for Milvus vector database client.

    Configuration parameters:
        - host: Hostname of Milvus server (default: "localhost").
        - port: Port number (default: 19530).
        - database: Database name (default: "default").
        - uri: Connection URI string (alternative to host:port).
        - username / password: Authentication credentials.
        - timeout: Connection timeout in seconds (default: 60).
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
            resolved_token = (
                f"{resolved_username}:{resolved_password}"
            )

        params = resolve_parameters(
            config,
            allowed_keys=MILVUS_ALLOWED_KEYS,
            aliases=MILVUS_PARAM_MAP,
            uri=resolved_uri,
            db_name=database,
            token=resolved_token,
            timeout=timeout,
            **kwargs,
        )

        self.client = MilvusClient(**params)

    # -- Connection --------------------------------------------------

    def connect(self) -> None:
        """Establish connection to Milvus.

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

    # -- Collection CRUD ---------------------------------------------

    def create_collection(self, config: CollectionConfig) -> None:
        """Create a Milvus collection.

        Args:
            config: Collection configuration parameters.

        Raises:
            RuntimeError: If creation fails.
        """
        params: dict[str, Any] = {
            "collection_name": config.name,
        }
        if config.dimension is not None:
            params["dimension"] = config.dimension
        if config.metric is not None:
            params["metric_type"] = _METRIC_MAP.get(
                config.metric, "COSINE"
            )
        params.update(config.kwargs)
        try:
            self.client.create_collection(**params)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create Milvus collection "
                f"'{config.name}': {exc}"
            ) from exc

    def delete_collection(self, name: str) -> None:
        """Delete a Milvus collection.

        Args:
            name: Collection name.

        Raises:
            RuntimeError: If deletion fails.
        """
        try:
            self.client.drop_collection(collection_name=name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to delete Milvus collection "
                f"'{name}': {exc}"
            ) from exc

    def list_collections(self) -> list[str]:
        """Return sorted names of all Milvus collections.

        Returns:
            Sorted list of collection names.
        """
        return sorted(self.client.list_collections())

    def has_collection(self, name: str) -> bool:
        """Check whether a Milvus collection exists.

        Args:
            name: Collection name.

        Returns:
            True if the collection exists, False otherwise.
        """
        return self.client.has_collection(collection_name=name)

    def upsert(
        self,
        collection_name: str,
        records: list[VectorRecord],
        **kwargs: Any,
    ) -> None:
        """Insert or update vector records in a Milvus collection."""
        rows = [
            {"id": record.id, "vector": record.vector, **record.payload}
            for record in records
        ]
        self.client.upsert(
            collection_name=collection_name,
            data=rows,
            **kwargs,
        )

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 5,
        **kwargs: Any,
    ) -> list[VectorSearchResult]:
        """Run similarity search against a Milvus collection."""
        response = self.client.search(
            collection_name=collection_name,
            data=[query_vector],
            limit=limit,
            **kwargs,
        )
        hits = response[0] if response else []
        results: list[VectorSearchResult] = []
        for hit in hits:
            entity = hit.get("entity", {})
            payload = {k: v for k, v in entity.items() if k not in {"id", "vector"}}
            identifier = entity.get("id", hit.get("id", ""))
            score = hit.get("distance", hit.get("score", 0.0))
            results.append(
                VectorSearchResult(
                    id=str(identifier),
                    score=float(score),
                    payload=payload,
                )
            )
        return results

    def delete(
        self,
        collection_name: str,
        ids: list[str],
        **kwargs: Any,
    ) -> None:
        """Delete records by IDs from a Milvus collection."""
        self.client.delete(
            collection_name=collection_name,
            ids=ids,
            **kwargs,
        )
{%- endif -%}
