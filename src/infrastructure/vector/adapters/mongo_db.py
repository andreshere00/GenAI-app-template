from typing import Any, Optional

from pymongo import MongoClient

from ....domain.vector import (
    MONGO_PARAM_MAP,
    CollectionConfig,
    VectorDBConfig,
    VectorDBProvider,
)
from ...utils import resolve_parameters
from ..base import BaseVectorDatabase
from ..factory import VectorDBFactory

MONGO_ALLOWED_KEYS: set[str] = {
    "host",
    "port",
    "username",
    "password",
    "serverSelectionTimeoutMS",
    "authSource",
    "tls",
    "replicaSet",
}


@VectorDBFactory.register(VectorDBProvider.MONGODB)
class MongoDBVectorDatabase(BaseVectorDatabase):
    """Wrapper for MongoDB vector database client.

    Configuration parameters:
        - host: MongoDB server hostname.
        - port: Server port (default: 27017).
        - username / password: Authentication credentials.
        - database: Database name to connect to.
        - url: Full MongoDB connection string.
        - timeout: Connection timeout in milliseconds.
    """

    def __init__(
        self,
        config: Optional[VectorDBConfig] = None,
        *,
        host: Optional[str] = None,
        port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        url: Optional[str] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize MongoDB vector database adapter.

        Args:
            config: VectorDBConfig with connection parameters.
            host: MongoDB server hostname.
            port: MongoDB server port.
            username: Username for authentication.
            password: Password for authentication.
            database: Default database name.
            url: MongoDB connection string.
            timeout: Connection timeout in milliseconds.
            **kwargs: Additional PyMongo-specific parameters.
        """
        super().__init__(config, **kwargs)

        config_fields = resolve_parameters(
            config,
            allowed_keys={
                "url", "host", "port",
                "username", "password",
                "database", "timeout",
            },
        )

        cfg_url = url or config_fields.get("url")
        cfg_host = host or config_fields.get("host")
        cfg_port = port or config_fields.get("port")
        cfg_user = username or config_fields.get("username")
        cfg_password = password or config_fields.get("password")
        cfg_database = database or config_fields.get("database")
        cfg_timeout = timeout or config_fields.get("timeout")

        if (
            cfg_url is None
            and cfg_host is not None
            and cfg_user
            and cfg_password
        ):
            cfg_url = (
                f"mongodb+srv://{cfg_user}:{cfg_password}"
                f"@{cfg_host}/{cfg_database or ''}"
            )
        elif cfg_url is None and cfg_host is not None:
            cfg_url = (
                f"mongodb://{cfg_host}:{cfg_port or 27017}"
            )

        params = resolve_parameters(
            config,
            allowed_keys=MONGO_ALLOWED_KEYS,
            aliases=MONGO_PARAM_MAP,
            host=cfg_url,
            port=None if cfg_url else cfg_port,
            username=cfg_user,
            password=cfg_password,
            serverSelectionTimeoutMS=cfg_timeout,
            **kwargs,
        )

        self.client = MongoClient(**params)
        self.database_name: Optional[str] = cfg_database

    # -- Connection --------------------------------------------------

    def connect(self) -> None:
        """Establish connection to MongoDB.

        Raises:
            ConnectionError: If connection cannot be established.
        """
        try:
            self.client.admin.command("ping")
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to MongoDB: {e}"
            )

    def disconnect(self) -> None:
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()

    def health(self) -> bool:
        """Check if MongoDB connection is healthy.

        Returns:
            True if database is accessible, False otherwise.
        """
        try:
            self.client.admin.command("ping")
            return True
        except Exception:
            return False

    # -- Collection CRUD ---------------------------------------------

    def _get_database(self) -> Any:
        """Return the database handle for the configured database."""
        if not self.database_name:
            raise RuntimeError(
                "No database name configured for MongoDB."
            )
        return self.client[self.database_name]

    def create_collection(self, config: CollectionConfig) -> None:
        """Create a MongoDB collection.

        Args:
            config: Collection configuration. Pass additional
                options via ``config.kwargs``.

        Raises:
            RuntimeError: If creation fails.
        """
        try:
            db = self._get_database()
            db.create_collection(config.name, **config.kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create MongoDB collection "
                f"'{config.name}': {exc}"
            ) from exc

    def delete_collection(self, name: str) -> None:
        """Delete a MongoDB collection.

        Args:
            name: Collection name.

        Raises:
            RuntimeError: If deletion fails.
        """
        try:
            db = self._get_database()
            db.drop_collection(name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to delete MongoDB collection "
                f"'{name}': {exc}"
            ) from exc

    def list_collections(self) -> list[str]:
        """Return sorted names of all MongoDB collections.

        Returns:
            Sorted list of collection names.
        """
        db = self._get_database()
        return sorted(db.list_collection_names())

    def has_collection(self, name: str) -> bool:
        """Check whether a MongoDB collection exists.

        Args:
            name: Collection name.

        Returns:
            True if the collection exists, False otherwise.
        """
        return name in self.list_collections()
