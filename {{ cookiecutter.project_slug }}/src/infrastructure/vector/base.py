{%- if cookiecutter.vector_db -%}
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from typing_extensions import Self

from ...domain.vector import CollectionConfig, VectorDBConfig


class BaseVectorDatabase(ABC):
    """Abstract base class for vector database implementations.

    Subclasses must provide concrete implementations for connection
    lifecycle and collection CRUD operations. The context-manager
    protocol (``__enter__`` / ``__exit__``) enables Unit-of-Work style
    resource management.

    Attributes:
        client: The underlying client instance for the specific provider.
        config: Configuration object containing database connection
            parameters.
    """

    def __init__(
        self,
        config: Optional[VectorDBConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the vector database adapter.

        Args:
            config: Configuration object with database connection
                parameters.
            **kwargs: Additional provider-specific arguments that
                override config values.
        """
        self.config = config
        self.client: Any = None

    # -- Context-manager lifecycle -----------------------------------

    def __enter__(self) -> Self:
        """Enter the context manager by establishing a connection."""
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> None:
        """Exit the context manager by closing the connection."""
        self.disconnect()

    # -- Connection --------------------------------------------------

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the vector database.

        Raises:
            ConnectionError: If connection cannot be established.
        """

    @abstractmethod
    def disconnect(self) -> None:
        """Close the connection to the vector database."""

    @abstractmethod
    def health(self) -> bool:
        """Check if the database connection is healthy.

        Returns:
            True if database is accessible, False otherwise.
        """

    # -- Collection CRUD ---------------------------------------------

    @abstractmethod
    def create_collection(self, config: CollectionConfig) -> None:
        """Create a new collection / index.

        Args:
            config: Collection configuration parameters.

        Raises:
            RuntimeError: If the collection already exists or creation
                fails.
        """

    @abstractmethod
    def delete_collection(self, name: str) -> None:
        """Delete an existing collection / index.

        Args:
            name: Name of the collection to delete.

        Raises:
            RuntimeError: If deletion fails.
        """

    @abstractmethod
    def list_collections(self) -> list[str]:
        """Return the names of all collections / indexes.

        Returns:
            Sorted list of collection names.
        """

    @abstractmethod
    def has_collection(self, name: str) -> bool:
        """Check whether a collection / index exists.

        Args:
            name: Name of the collection to look up.

        Returns:
            True if the collection exists, False otherwise.
        """
{%- endif -%}
