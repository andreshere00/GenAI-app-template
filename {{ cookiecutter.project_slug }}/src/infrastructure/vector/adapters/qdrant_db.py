{%- if "6" in cookiecutter.vector_db -%}
from typing import Any, Optional
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointIdsList, PointStruct, VectorParams

from ....domain.vector import (
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

QDRANT_ALLOWED_KEYS: set[str] = {
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

_METRIC_MAP: dict[str, Distance] = {
    DistanceMetric.COSINE: Distance.COSINE,
    DistanceMetric.EUCLIDEAN: Distance.EUCLID,
    DistanceMetric.DOT_PRODUCT: Distance.DOT,
}


@VectorDBFactory.register(VectorDBProvider.QDRANT)
class QdrantVectorDatabase(BaseVectorDatabase):
    """Wrapper for Qdrant vector database client.

    Configuration parameters:
        - url: HTTP endpoint URL (e.g., "http://localhost:6333").
        - api_key: API key for cloud deployments.
        - host / port: Address for local or self-hosted instances.
        - grpc_port: gRPC port for high-performance connections.
        - prefer_grpc: Use gRPC instead of REST.
        - https: Use HTTPS for connections.
        - prefix: URL path prefix.
        - timeout: Request timeout in seconds.
        - path: Path for in-memory / persisted local mode.
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

    # -- Connection --------------------------------------------------

    def connect(self) -> None:
        """Establish connection to Qdrant.

        Raises:
            ConnectionError: If connection cannot be established.
        """
        try:
            self.client.get_collections()
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to Qdrant: {e}"
            )

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
            self.client.get_collections()
            return True
        except Exception:
            return False

    @staticmethod
    def _coerce_point_id(point_id: str) -> int | UUID:
        """Coerce string IDs into Qdrant-supported point ID types.

        Qdrant point IDs must be either an unsigned 64-bit integer or a UUID.
        """
        if point_id.isdigit():
            value = int(point_id)
            if value < 0 or value > 2**64 - 1:
                raise ValueError(
                    "Qdrant numeric point IDs must fit in unsigned 64-bit range."
                )
            return value
        try:
            return UUID(point_id)
        except ValueError as exc:
            raise ValueError(
                "Qdrant point IDs must be a UUID string or a numeric string."
            ) from exc

    # -- Collection CRUD ---------------------------------------------

    def create_collection(self, config: CollectionConfig) -> None:
        """Create a Qdrant collection.

        Args:
            config: Collection configuration. ``dimension`` is required.

        Raises:
            RuntimeError: If creation fails.
        """
        distance = _METRIC_MAP.get(
            config.metric, Distance.COSINE
        ) if config.metric else Distance.COSINE
        vectors_config = VectorParams(
            size=config.dimension or 0,
            distance=distance,
        )
        extra: dict[str, Any] = dict(config.kwargs)
        try:
            self.client.create_collection(
                collection_name=config.name,
                vectors_config=vectors_config,
                **extra,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create Qdrant collection "
                f"'{config.name}': {exc}"
            ) from exc

    def delete_collection(self, name: str) -> None:
        """Delete a Qdrant collection.

        Args:
            name: Collection name.

        Raises:
            RuntimeError: If deletion fails.
        """
        try:
            self.client.delete_collection(collection_name=name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to delete Qdrant collection "
                f"'{name}': {exc}"
            ) from exc

    def list_collections(self) -> list[str]:
        """Return sorted names of all Qdrant collections.

        Returns:
            Sorted list of collection names.
        """
        response = self.client.get_collections()
        return sorted(c.name for c in response.collections)

    def has_collection(self, name: str) -> bool:
        """Check whether a Qdrant collection exists.

        Args:
            name: Collection name.

        Returns:
            True if the collection exists, False otherwise.
        """
        return self.client.collection_exists(
            collection_name=name
        )

    def upsert(
        self,
        collection_name: str,
        records: list[VectorRecord],
        **kwargs: Any,
    ) -> None:
        """Insert or update vector records in a Qdrant collection."""
        points = [
            PointStruct(
                id=self._coerce_point_id(record.id),
                vector=record.vector,
                payload=record.payload,
            )
            for record in records
        ]
        self.client.upsert(
            collection_name=collection_name,
            points=points,
            **kwargs,
        )

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 5,
        **kwargs: Any,
    ) -> list[VectorSearchResult]:
        """Run similarity search against a Qdrant collection."""
        response = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            **kwargs,
        )
        return [
            VectorSearchResult(
                id=str(point.id),
                score=float(point.score),
                payload=dict(point.payload or {}),
            )
            for point in response
        ]

    def delete(
        self,
        collection_name: str,
        ids: list[str],
        **kwargs: Any,
    ) -> None:
        """Delete records by IDs from a Qdrant collection."""
        self.client.delete(
            collection_name=collection_name,
            points_selector=PointIdsList(
                points=[self._coerce_point_id(i) for i in ids]
            ),
            **kwargs,
        )
{%- endif -%}
