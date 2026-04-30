import math
from typing import Any, Optional

from azure.cosmos import CosmosClient

from ....domain.vector import (
    COSMOS_PARAM_MAP,
    CollectionConfig,
    VectorRecord,
    VectorSearchResultDTO as VectorSearchResult,
    VectorDBConfig,
    VectorDBProvider,
)
from ...utils import resolve_parameters
from ..base import BaseVectorDatabase
from ..factory import VectorDBFactory

COSMOS_ALLOWED_KEYS: set[str] = {
    "url",
    "credential",
    "connection_timeout",
    "connection_policy",
    "consistency_level",
}


@VectorDBFactory.register(VectorDBProvider.COSMOS_DB)
class CosmosDBVectorDatabase(BaseVectorDatabase):
    """Wrapper for Azure Cosmos DB vector database client.

    Configuration parameters:
        - url: Cosmos DB account endpoint URL (required).
        - database: Database name (required).
        - api_key: Account primary or secondary key (required).
        - timeout: Request timeout in seconds.
        - connection_policy: Custom connection policy dict.
        - max_retries: Maximum number of retry attempts.
    """

    def __init__(
        self,
        config: Optional[VectorDBConfig] = None,
        *,
        url: Optional[str] = None,
        database: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
        connection_policy: Optional[dict[str, Any]] = None,
        max_retries: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Cosmos DB vector database adapter.

        Args:
            config: VectorDBConfig with connection parameters.
            url: Cosmos DB account endpoint URL.
            database: Database name.
            api_key: Cosmos DB account key.
            timeout: Request timeout in seconds.
            connection_policy: Custom connection policy configuration.
            max_retries: Maximum retry attempts.
            **kwargs: Additional Cosmos DB-specific parameters.
        """
        super().__init__(config, **kwargs)

        config_fields = resolve_parameters(
            config,
            allowed_keys={"api_key", "database"},
        )
        cfg_api_key = api_key or config_fields.get("api_key")
        cfg_database = database or config_fields.get("database")

        params = resolve_parameters(
            config,
            allowed_keys=COSMOS_ALLOWED_KEYS,
            aliases=COSMOS_PARAM_MAP,
            url=url,
            credential=cfg_api_key,
            connection_timeout=timeout or 60,
            **kwargs,
        )

        if connection_policy:
            params["connection_policy"] = connection_policy
        if max_retries is not None:
            params["max_retries"] = max_retries

        self.client = CosmosClient(**params)
        self.database_name: Optional[str] = cfg_database

    # -- Connection --------------------------------------------------

    def connect(self) -> None:
        """Establish connection to Cosmos DB.

        Raises:
            ConnectionError: If connection cannot be established.
        """
        try:
            if self.database_name:
                self.client.get_database_client(
                    self.database_name
                )
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to Cosmos DB database "
                f"{self.database_name}: {e}"
            )

    def disconnect(self) -> None:
        """Close the Cosmos DB connection."""
        if self.client:
            self.client.close()

    def health(self) -> bool:
        """Check if Cosmos DB connection is healthy.

        Returns:
            True if database is accessible, False otherwise.
        """
        try:
            if self.database_name:
                db = self.client.get_database_client(
                    self.database_name
                )
                return db is not None
            return True
        except Exception:
            return False

    # -- Collection CRUD ---------------------------------------------

    def _get_database(self) -> Any:
        """Return the database proxy for the configured database."""
        if not self.database_name:
            raise RuntimeError(
                "No database name configured for Cosmos DB."
            )
        return self.client.get_database_client(self.database_name)

    def create_collection(self, config: CollectionConfig) -> None:
        """Create a Cosmos DB container.

        Args:
            config: Collection configuration. Pass
                ``partition_key`` via ``config.kwargs``.

        Raises:
            RuntimeError: If creation fails.
        """
        extra: dict[str, Any] = dict(config.kwargs)
        partition_key = extra.pop("partition_key", "/id")
        try:
            db = self._get_database()
            db.create_container(
                id=config.name,
                partition_key=partition_key,
                **extra,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create Cosmos DB container "
                f"'{config.name}': {exc}"
            ) from exc

    def delete_collection(self, name: str) -> None:
        """Delete a Cosmos DB container.

        Args:
            name: Container name.

        Raises:
            RuntimeError: If deletion fails.
        """
        try:
            db = self._get_database()
            db.delete_container(name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to delete Cosmos DB container "
                f"'{name}': {exc}"
            ) from exc

    def list_collections(self) -> list[str]:
        """Return sorted names of all Cosmos DB containers.

        Returns:
            Sorted list of container names.
        """
        db = self._get_database()
        return sorted(c["id"] for c in db.list_containers())

    def has_collection(self, name: str) -> bool:
        """Check whether a Cosmos DB container exists.

        Args:
            name: Container name.

        Returns:
            True if the container exists, False otherwise.
        """
        return name in self.list_collections()

    @staticmethod
    def _cosine_similarity(first: list[float], second: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(first) != len(second) or not first:
            return 0.0
        dot = sum(a * b for a, b in zip(first, second))
        norm_first = math.sqrt(sum(a * a for a in first))
        norm_second = math.sqrt(sum(b * b for b in second))
        if norm_first == 0.0 or norm_second == 0.0:
            return 0.0
        return dot / (norm_first * norm_second)

    @staticmethod
    def _validate_field_name(field_name: str) -> str:
        """Validate a Cosmos SQL property name used in vector search queries."""
        if not field_name or not field_name.replace("_", "").isalnum():
            raise ValueError(
                "Invalid vector field name. Use only letters, digits, and underscores."
            )
        return field_name

    # -- Vector CRUD ---------------------------------------------

    def upsert(
        self,
        collection_name: str,
        records: list[VectorRecord],
        **kwargs: Any,
    ) -> None:
        """Insert or update vector records in a Cosmos DB container."""
        db = self._get_database()
        container = db.get_container_client(collection_name)
        for record in records:
            item = {
                "id": record.id,
                "vector": record.vector,
                "payload": record.payload,
            }
            container.upsert_item(item, **kwargs)

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 5,
        **kwargs: Any,
    ) -> list[VectorSearchResult]:
        """Run vector search in Cosmos DB using VectorDistance().

        This requires Cosmos DB for NoSQL vector search to be enabled and a
        vector index policy configured on the container.
        """
        db = self._get_database()
        container = db.get_container_client(collection_name)
        vector_field = self._validate_field_name(
            kwargs.pop("vector_field", "vector")
        )
        enable_fallback = bool(
            kwargs.pop("fallback_client_side", False)
        )

        query = (
            f"SELECT TOP {limit} c.id, c.payload, "
            f"VectorDistance(c.{vector_field}, @embedding) AS score "
            f"FROM c ORDER BY VectorDistance(c.{vector_field}, @embedding)"
        )
        try:
            items = container.query_items(
                query=query,
                parameters=[
                    {"name": "@embedding", "value": query_vector}
                ],
                enable_cross_partition_query=True,
                **kwargs,
            )
            return [
                VectorSearchResult(
                    id=str(item.get("id", "")),
                    score=float(item.get("score", 0.0)),
                    payload=dict(item.get("payload", {}) or {}),
                )
                for item in items
            ]
        except Exception:
            if not enable_fallback:
                raise

        items = container.query_items(
            query="SELECT c.id, c.payload, c.vector FROM c",
            enable_cross_partition_query=True,
            **kwargs,
        )
        scored: list[VectorSearchResult] = []
        for item in items:
            vector = item.get(vector_field, item.get("vector", []))
            score = self._cosine_similarity(query_vector, vector)
            scored.append(
                VectorSearchResult(
                    id=str(item.get("id", "")),
                    score=score,
                    payload=dict(item.get("payload", {}) or {}),
                )
            )
        return sorted(scored, key=lambda hit: hit.score, reverse=True)[:limit]

    def delete(
        self,
        collection_name: str,
        ids: list[str],
        **kwargs: Any,
    ) -> None:
        """Delete records by IDs from a Cosmos DB container."""
        db = self._get_database()
        container = db.get_container_client(collection_name)
        partition_key_field = kwargs.pop("partition_key_field", "id")
        for record_id in ids:
            partition_key = kwargs.get("partition_key", record_id)
            container.delete_item(
                item={partition_key_field: record_id, "id": record_id},
                partition_key=partition_key,
            )
