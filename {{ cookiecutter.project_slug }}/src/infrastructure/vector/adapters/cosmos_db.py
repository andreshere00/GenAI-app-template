{%- if "1" in cookiecutter.vector_db -%}
from typing import Any, Optional

from azure.cosmos import CosmosClient

from ....domain.vector import VectorDBConfig
from ...utils import resolve_parameters
from ..base import BaseVectorDatabase

COSMOS_ALLOWED_KEYS = {
    "url",
    "credential",
    "connection_timeout",
    "connection_policy",
    "consistency_level",
}


class CosmosDBVectorDatabase(BaseVectorDatabase):
    """Wrapper for Azure Cosmos DB vector database client.

    Azure Cosmos DB provides vector similarity search capabilities. This
    adapter manages connections to Cosmos DB accounts with the NoSQL API.

    Configuration parameters:
        - url: Cosmos DB account endpoint URL (required, starts with https://)
        - database: Database name (required)
        - api_key: Account primary or secondary key (required)
        - timeout: Request timeout in seconds
        - connection_policy: Custom connection policy dict
        - max_retries: Maximum number of retry attempts
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
        config_api_key = config_fields.get("api_key")
        config_database = config_fields.get("database")

        cfg_api_key = api_key or config_api_key
        cfg_database = database or config_database

        params = resolve_parameters(
            config,
            allowed_keys=COSMOS_ALLOWED_KEYS,
            aliases={"timeout": "connection_timeout", "api_key": "credential"},
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
        self.database_name = cfg_database

    def connect(self) -> None:
        """Establish connection to Cosmos DB.

        The CosmosClient constructor establishes the connection.
        This method validates connectivity by accessing the database.
        """
        try:
            if self.database_name:
                self.client.get_database_client(self.database_name)
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
                db = self.client.get_database_client(self.database_name)
                return db is not None
            return True
        except Exception:
            return False
{%- endif -%}
