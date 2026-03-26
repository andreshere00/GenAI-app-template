from abc import ABC, abstractmethod
from typing import Any, Optional

from ...domain.vector import VectorDBConfig


class BaseVectorDatabase(ABC):
    """Abstract base class for vector database implementations.

    This class defines the common interface that all vector database adapters
    must implement. Subclasses should provide concrete implementations for
    different providers (Qdrant, Pinecone, Milvus, etc.).

    Attributes:
        client: The underlying client instance for the specific provider.
        config: Configuration object containing database connection parameters.
    """

    def __init__(
        self,
        config: Optional[VectorDBConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the vector database adapter.

        Args:
            config: Configuration object with database connection parameters.
            **kwargs: Additional provider-specific arguments that override
                config values.
        """
        self.config = config
        self.client: Any = None

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
