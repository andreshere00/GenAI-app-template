{%- if "7" in cookiecutter.vector_db -%}
from typing import Any, Optional

from google.cloud import aiplatform

from ....domain.vector import (
    VERTEX_PARAM_MAP,
    CollectionConfig,
    VectorRecord,
    VectorSearchResultDTO as VectorSearchResult,
    VectorDBConfig,
    VectorDBProvider,
)
from ...utils import resolve_parameters
from ..base import BaseVectorDatabase
from ..factory import VectorDBFactory

VERTEX_ALLOWED_KEYS: set[str] = {
    "project_id",
    "region",
    "index_name",
    "credentials",
    "api_key",
    "timeout",
}


@VectorDBFactory.register(VectorDBProvider.VERTEX_AI)
class VertexDBVectorDatabase(BaseVectorDatabase):
    """Wrapper for Google Vertex AI Vector Search client.

    Configuration parameters:
        - project_id: Google Cloud Project ID (required).
        - region: GCP region (default: "us-central1").
        - index_name: Name of the vector search index.
        - credentials: Path to service account JSON key.
        - api_key: Google Cloud API key (alternative to credentials).
        - timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        config: Optional[VectorDBConfig] = None,
        *,
        project_id: Optional[str] = None,
        region: Optional[str] = None,
        index_name: Optional[str] = None,
        credentials: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Google Vertex AI Vector Search adapter.

        Args:
            config: VectorDBConfig with connection parameters.
            project_id: Google Cloud Project ID.
            region: GCP region for the service.
            index_name: Name of the vector search index.
            credentials: Path to service account credentials.
            api_key: Google API key for authentication.
            timeout: Request timeout in seconds.
            **kwargs: Additional Vertex AI-specific parameters.
        """
        super().__init__(config, **kwargs)

        params = resolve_parameters(
            config,
            allowed_keys=VERTEX_ALLOWED_KEYS,
            aliases=VERTEX_PARAM_MAP,
            project_id=project_id,
            region=region,
            index_name=index_name,
            credentials=credentials,
            api_key=api_key,
            timeout=timeout,
            **kwargs,
        )

        self.project_id: Optional[str] = params.get("project_id")
        self.region: str = params.get("region") or "us-central1"
        self.index_name: Optional[str] = params.get("index_name")
        self.credentials: Optional[str] = params.get(
            "credentials"
        )
        self.api_key: Optional[str] = params.get("api_key")
        self.timeout: Optional[int] = params.get("timeout")
        self._initialized: bool = False
        self._memory_store: dict[str, dict[str, VectorRecord]] = {}

    def _init_client_context(self) -> None:
        """Initialize the Vertex AI SDK context."""
        init_kwargs: dict[str, Any] = {
            "project": self.project_id,
            "location": self.region,
        }
        if self.credentials is not None:
            init_kwargs["credentials"] = self.credentials
        if self.api_key is not None:
            init_kwargs["api_key"] = self.api_key
        if self.timeout is not None:
            init_kwargs["request_timeout"] = self.timeout
        aiplatform.init(**init_kwargs)
        self._initialized = True

    # -- Connection --------------------------------------------------

    def connect(self) -> None:
        """Establish connection to Vertex AI Vector Search.

        Raises:
            ConnectionError: If connection cannot be established.
        """
        try:
            if not self._initialized:
                self._init_client_context()
            if self.index_name:
                index = aiplatform.MatchingEngineIndex(
                    index_name=self.index_name,
                    project=self.project_id,
                    location=self.region,
                )
                self.client = index
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to Vertex AI index "
                f"{self.index_name}: {e}"
            )

    def disconnect(self) -> None:
        """Close the Vertex AI connection.

        Vertex AI automatically manages connections.
        """

    def health(self) -> bool:
        """Check if Vertex AI connection is healthy.

        Returns:
            True if service is accessible, False otherwise.
        """
        try:
            if not self._initialized:
                self._init_client_context()
            if self.index_name:
                index = aiplatform.MatchingEngineIndex(
                    index_name=self.index_name,
                    project=self.project_id,
                    location=self.region,
                )
                return index is not None
            return True
        except Exception:
            return False

    # -- Collection CRUD ---------------------------------------------

    def create_collection(self, config: CollectionConfig) -> None:
        """Create a Vertex AI Matching Engine index.

        Args:
            config: Collection configuration. ``dimension`` is
                required. Pass additional tree-AH parameters via
                ``config.kwargs``.

        Raises:
            RuntimeError: If creation fails.
        """
        if not self._initialized:
            self._init_client_context()
        extra: dict[str, Any] = dict(config.kwargs)
        try:
            aiplatform.MatchingEngineIndex.create_tree_ah_index(
                display_name=config.name,
                dimensions=config.dimension or 0,
                distance_measure_type=config.metric or "COSINE",
                project=self.project_id,
                location=self.region,
                **extra,
            )
            self._memory_store.setdefault(config.name, {})
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create Vertex AI index "
                f"'{config.name}': {exc}"
            ) from exc

    def delete_collection(self, name: str) -> None:
        """Delete a Vertex AI Matching Engine index.

        Args:
            name: Index display name.

        Raises:
            RuntimeError: If deletion fails.
        """
        if not self._initialized:
            self._init_client_context()
        try:
            indexes = aiplatform.MatchingEngineIndex.list(
                project=self.project_id,
                location=self.region,
            )
            for idx in indexes:
                if idx.display_name == name:
                    idx.delete()
                    self._memory_store.pop(name, None)
                    return
            raise RuntimeError(
                f"Vertex AI index '{name}' not found."
            )
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Failed to delete Vertex AI index "
                f"'{name}': {exc}"
            ) from exc

    def list_collections(self) -> list[str]:
        """Return sorted display names of all Vertex AI indexes.

        Returns:
            Sorted list of index display names.
        """
        if not self._initialized:
            self._init_client_context()
        indexes = aiplatform.MatchingEngineIndex.list(
            project=self.project_id,
            location=self.region,
        )
        names = {idx.display_name for idx in indexes}
        names.update(self._memory_store.keys())
        return sorted(names)

    def has_collection(self, name: str) -> bool:
        """Check whether a Vertex AI index exists.

        Args:
            name: Index display name.

        Returns:
            True if the index exists, False otherwise.
        """
        return name in self.list_collections()

    def upsert(
        self,
        collection_name: str,
        records: list[VectorRecord],
        **kwargs: Any,
    ) -> None:
        """Insert or update vector records in a local Vertex AI cache."""
        collection = self._memory_store.setdefault(collection_name, {})
        for record in records:
            collection[record.id] = record

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 5,
        **kwargs: Any,
    ) -> list[VectorSearchResult]:
        """Run vector similarity search against the local Vertex cache."""
        records = list(self._memory_store.get(collection_name, {}).values())
        scored = []
        for record in records:
            score = self._cosine_similarity(query_vector, record.vector)
            scored.append(
                VectorSearchResult(
                    id=record.id,
                    score=score,
                    payload=dict(record.payload),
                )
            )
        return sorted(scored, key=lambda hit: hit.score, reverse=True)[:limit]

    def delete(
        self,
        collection_name: str,
        ids: list[str],
        **kwargs: Any,
    ) -> None:
        """Delete records by IDs from the local Vertex cache."""
        collection = self._memory_store.setdefault(collection_name, {})
        for record_id in ids:
            collection.pop(record_id, None)

    @staticmethod
    def _cosine_similarity(first: list[float], second: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(first) != len(second) or not first:
            return 0.0
        dot = sum(a * b for a, b in zip(first, second))
        norm_first = sum(a * a for a in first) ** 0.5
        norm_second = sum(b * b for b in second) ** 0.5
        if norm_first == 0.0 or norm_second == 0.0:
            return 0.0
        return dot / (norm_first * norm_second)
{%- endif -%}
