{%- if cookiecutter.vector_db -%}
from __future__ import annotations

from typing import Any, Optional, Protocol

from typing_extensions import Self


class VectorDBConfig(Protocol):
    """Protocol defining the configuration shape for vector databases.

    This protocol is extensible. Generic fields like 'database' or 'url'
    can be mapped to provider-specific arguments (e.g., 'db_name', 'uri')
    via adapter parameter maps.
    """

    host: Optional[str]
    port: Optional[int]
    url: Optional[str]
    api_key: Optional[str]
    grpc_port: Optional[int]
    https: Optional[bool]
    prefix: Optional[str]
    timeout: Optional[int]
    username: Optional[str]
    password: Optional[str]
    database: Optional[str]
    collection: Optional[str]
    region: Optional[str]
    kwargs: dict[str, Any]


class CollectionConfig(Protocol):
    """Protocol defining the configuration shape for a vector collection.

    Adapters translate these standardised fields into provider-specific
    SDK arguments (e.g., 'metric' -> Qdrant 'distance', Milvus
    'metric_type').
    """

    name: str
    dimension: Optional[int]
    metric: Optional[str]
    kwargs: dict[str, Any]


class VectorRecord(Protocol):
    """Protocol for a vector record stored in a collection."""

    id: str
    vector: list[float]
    payload: dict[str, Any]


class VectorSearchResult(Protocol):
    """Protocol for a vector search hit."""

    id: str
    score: float
    payload: dict[str, Any]


class VectorDB(Protocol):
    """Protocol declaring the full interface of a vector database wrapper.

    Covers connection lifecycle (including context-manager support) and
    collection CRUD operations.
    """

    config: Optional[VectorDBConfig]
    client: Any

    def connect(self) -> None:
        """Establish connection to the vector database.

        Raises:
            ConnectionError: If connection cannot be established.
        """
        ...

    def disconnect(self) -> None:
        """Close the connection to the vector database."""
        ...

    def health(self) -> bool:
        """Check if the database connection is healthy.

        Returns:
            True if database is accessible, False otherwise.
        """
        ...

    def __enter__(self) -> Self:
        """Enter the context manager (connect)."""
        ...

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> None:
        """Exit the context manager (disconnect)."""
        ...

    def create_collection(self, config: CollectionConfig) -> None:
        """Create a new collection / index.

        Args:
            config: Collection configuration parameters.

        Raises:
            RuntimeError: If the collection already exists or creation
                fails.
        """
        ...

    def delete_collection(self, name: str) -> None:
        """Delete an existing collection / index.

        Args:
            name: Name of the collection to delete.

        Raises:
            RuntimeError: If deletion fails.
        """
        ...

    def list_collections(self) -> list[str]:
        """Return the names of all collections / indexes.

        Returns:
            Sorted list of collection names.
        """
        ...

    def has_collection(self, name: str) -> bool:
        """Check whether a collection / index exists.

        Args:
            name: Name of the collection to look up.

        Returns:
            True if the collection exists, False otherwise.
        """
        ...

    def upsert(
        self,
        collection_name: str,
        records: list[VectorRecord],
        **kwargs: Any,
    ) -> None:
        """Insert or update vector records in a collection.

        Args:
            collection_name: Target collection/index name.
            records: Records containing id, vector and payload.
            **kwargs: Provider-specific write options.
        """
        ...

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 5,
        **kwargs: Any,
    ) -> list[VectorSearchResult]:
        """Run a vector similarity search.

        Args:
            collection_name: Target collection/index name.
            query_vector: Query embedding vector.
            limit: Maximum number of hits to return.
            **kwargs: Provider-specific search options.

        Returns:
            Ranked list of search results.
        """
        ...

    def delete(
        self,
        collection_name: str,
        ids: list[str],
        **kwargs: Any,
    ) -> None:
        """Delete vector records by IDs.

        Args:
            collection_name: Target collection/index name.
            ids: Record identifiers to delete.
            **kwargs: Provider-specific delete options.
        """
        ...
{%- endif -%}
