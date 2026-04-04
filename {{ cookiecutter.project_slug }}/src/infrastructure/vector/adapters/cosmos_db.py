{%- if "1" in cookiecutter.vector_db -%}
from typing import Any, Optional

from azure.cosmos import CosmosClient

from ....domain.vector import (
    COSMOS_PARAM_MAP,
    CollectionConfig,
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
{%- endif -%}
