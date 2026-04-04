{%- if "5" in cookiecutter.vector_db -%}
from typing import Any, Optional

from pinecone import Pinecone, ServerlessSpec

from ....domain.vector import (
    CollectionConfig,
    VectorDBConfig,
    VectorDBProvider,
)
from ...utils import resolve_parameters
from ..base import BaseVectorDatabase
from ..factory import VectorDBFactory

PINECONE_CLIENT_ALLOWED_KEYS: set[str] = {"api_key", "host"}


@VectorDBFactory.register(VectorDBProvider.PINECONE)
class PineconeVectorDatabase(BaseVectorDatabase):
    """Wrapper for Pinecone vector database client.

    Configuration parameters:
        - api_key: Pinecone API key (required).
        - host: Pinecone environment / host.
        - index_name: Name of the index to use.
        - timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        config: Optional[VectorDBConfig] = None,
        *,
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        index_name: Optional[str] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Pinecone vector database adapter.

        Args:
            config: VectorDBConfig with connection parameters.
            api_key: Pinecone API key for authentication.
            host: Pinecone host / environment.
            index_name: Name of the Pinecone index to target.
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
        self.index_name: Optional[str] = (
            index_name or config_collection
        )
        self.host: Optional[str] = params.get("host")

    # -- Connection --------------------------------------------------

    def connect(self) -> None:
        """Establish connection to Pinecone.

        Raises:
            ConnectionError: If connection cannot be established.
        """
        if self.index_name:
            try:
                self.client.Index(self.index_name)
            except Exception as e:
                raise ConnectionError(
                    f"Failed to connect to Pinecone index "
                    f"{self.index_name}: {e}"
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
                idx_kwargs: dict[str, Any] = {}
                if self.host:
                    idx_kwargs["host"] = self.host
                index = self.client.Index(
                    self.index_name, **idx_kwargs
                )
                return index.describe_index_stats() is not None
            return True
        except Exception:
            return False

    # -- Collection CRUD ---------------------------------------------

    def create_collection(self, config: CollectionConfig) -> None:
        """Create a Pinecone index.

        Args:
            config: Collection configuration. ``dimension`` is
                required. Pass ``spec`` via ``config.kwargs`` to
                control pod / serverless placement.

        Raises:
            RuntimeError: If creation fails.
        """
        extra: dict[str, Any] = dict(config.kwargs)
        spec = extra.pop("spec", ServerlessSpec(
            cloud="aws", region="us-east-1",
        ))
        try:
            self.client.create_index(
                name=config.name,
                dimension=config.dimension or 0,
                metric=config.metric or "cosine",
                spec=spec,
                **extra,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create Pinecone index "
                f"'{config.name}': {exc}"
            ) from exc

    def delete_collection(self, name: str) -> None:
        """Delete a Pinecone index.

        Args:
            name: Index name.

        Raises:
            RuntimeError: If deletion fails.
        """
        try:
            self.client.delete_index(name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to delete Pinecone index "
                f"'{name}': {exc}"
            ) from exc

    def list_collections(self) -> list[str]:
        """Return sorted names of all Pinecone indexes.

        Returns:
            Sorted list of index names.
        """
        return sorted(
            idx.name for idx in self.client.list_indexes()
        )

    def has_collection(self, name: str) -> bool:
        """Check whether a Pinecone index exists.

        Args:
            name: Index name.

        Returns:
            True if the index exists, False otherwise.
        """
        return name in self.list_collections()
{%- endif -%}
