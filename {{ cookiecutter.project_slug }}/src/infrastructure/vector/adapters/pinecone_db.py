{%- if "5" in cookiecutter.vector_db -%}
from typing import Any, Optional

from pinecone import Pinecone

from ....domain.vector import VectorDBConfig
from ...utils import resolve_parameters
from ..base import BaseVectorDatabase

PINECONE_CLIENT_ALLOWED_KEYS = {"api_key", "host"}


class PineconeVectorDatabase(BaseVectorDatabase):
    """Wrapper for Pinecone vector database client.

    Pinecone is a fully managed vector database service with high availability
    and automatic scaling. This adapter manages connections to Pinecone
    indexes.

    Configuration parameters:
        - api_key: Pinecone API key (required)
        - host: Pinecone environment/host (e.g., "us-east1-aws")
        - index_name: Name of the index to use
        - environment: Pinecone environment (deprecated in newer versions)
        - timeout: Request timeout in seconds
    """

    def __init__(
        self,
        config: Optional[VectorDBConfig] = None,
        *,
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        index_name: Optional[str] = None,
        environment: Optional[str] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Pinecone vector database adapter.

        Args:
            config: VectorDBConfig with connection parameters.
            api_key: Pinecone API key for authentication.
            host: Pinecone host/environment.
            index_name: Name of the Pinecone index to target.
            environment: Pinecone environment (deprecated).
            timeout: Request timeout in seconds.
            **kwargs: Additional Pinecone-specific parameters.
        """
        super().__init__(config, **kwargs)

        params = resolve_parameters(
            config,
            allowed_keys=PINECONE_CLIENT_ALLOWED_KEYS,
            api_key=api_key,
            host=host,
            timeout=timeout,
            **kwargs,
        )

        self.client = Pinecone(**params)
        config_collection = resolve_parameters(
            config,
            allowed_keys={"collection"},
        ).get("collection")
        self.index_name = index_name or config_collection
        self.host = params.get("host")

    def connect(self) -> None:
        """Establish connection to Pinecone.

        The Pinecone client constructor establishes the connection.
        This method gets the target index to validate connectivity.
        """
        if self.index_name:
            try:
                self.client.Index(self.index_name)
            except Exception as e:
                raise ConnectionError(
                    f"Failed to connect to Pinecone index {self.index_name}: {e}"
                )

    def disconnect(self) -> None:
        """Close the Pinecone connection.

        Pinecone client handles connection cleanup automatically.
        """

    def health(self) -> bool:
        """Check if Pinecone connection is healthy.

        Returns:
            True if database is accessible, False otherwise.
        """
        try:
            if self.index_name:
                index_kwargs: dict[str, Any] = {}
                if self.host:
                    index_kwargs["host"] = self.host
                index = self.client.Index(self.index_name, **index_kwargs)
                return index.describe_index_stats() is not None
            return True
        except Exception:
            return False
{%- endif -%}
