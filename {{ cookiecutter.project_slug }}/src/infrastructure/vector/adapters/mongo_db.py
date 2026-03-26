{%- if "3" in cookiecutter.vector_db -%}
from typing import Any, Optional

from pymongo import MongoClient

from ....domain.vector import VectorDBConfig
from ...utils import resolve_parameters
from ..base import BaseVectorDatabase

MONGO_ALLOWED_KEYS = {
    "host",
    "port",
    "username",
    "password",
    "serverSelectionTimeoutMS",
    "authSource",
    "tls",
    "replicaSet",
}


class MongoDBVectorDatabase(BaseVectorDatabase):
    """Wrapper for MongoDB vector database client.

    MongoDB with Atlas Vector Search provides vector similarity search
    capabilities. This adapter manages connections to MongoDB instances,
    including MongoDB Atlas.

    Configuration parameters:
        - host: MongoDB server hostname (e.g., "localhost")
        - port: MongoDB server port (default: 27017)
        - username: Username for authentication
        - password: Password for authentication
        - database: Database name to connect to
        - url: Full MongoDB connection string (alternative to host:port)
        - timeout: Connection timeout in milliseconds
        - serverSelectionTimeoutMS: Server selection timeout in milliseconds
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
            url: MongoDB connection string (mongodb:// or mongodb+srv://).
            timeout: Connection timeout in milliseconds.
            **kwargs: Additional PyMongo-specific parameters.
        """
        super().__init__(config, **kwargs)

        config_fields = resolve_parameters(
            config,
            allowed_keys={
                "url",
                "host",
                "port",
                "username",
                "password",
                "database",
                "timeout",
            },
        )

        config_url = config_fields.get("url")
        config_host = config_fields.get("host")
        config_port = config_fields.get("port")
        config_user = config_fields.get("username")
        config_password = config_fields.get("password")
        config_database = config_fields.get("database")
        config_timeout = config_fields.get("timeout")

        cfg_url = url or config_url
        cfg_host = host or config_host
        cfg_port = port or config_port
        cfg_user = username or config_user
        cfg_password = password or config_password
        cfg_database = database or config_database
        cfg_timeout = timeout or config_timeout

        if cfg_url is None and cfg_host is not None and cfg_user and cfg_password:
            cfg_url = (
                f"mongodb+srv://{cfg_user}:{cfg_password}@{cfg_host}"
                f"/{cfg_database or ''}"
            )
        elif cfg_url is None and cfg_host is not None:
            cfg_url = f"mongodb://{cfg_host}:{cfg_port or 27017}"

        params = resolve_parameters(
            config,
            allowed_keys=MONGO_ALLOWED_KEYS,
            aliases={"timeout": "serverSelectionTimeoutMS"},
            host=cfg_url,
            port=None if cfg_url else cfg_port,
            username=cfg_user,
            password=cfg_password,
            serverSelectionTimeoutMS=cfg_timeout,
            **kwargs,
        )

        self.client = MongoClient(**params)
        self.database_name = cfg_database

    def connect(self) -> None:
        """Establish connection to MongoDB.

        The MongoClient constructor establishes the connection lazily.
        This method forces connection validation.
        """
        try:
            self.client.admin.command("ping")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to MongoDB: {e}")

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
{%- endif -%}
