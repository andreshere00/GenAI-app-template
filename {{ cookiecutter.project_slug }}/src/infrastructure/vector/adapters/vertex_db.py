{%- if "7" in cookiecutter.vector_db -%}
from typing import Any, Optional

from google.cloud import aiplatform
from google.cloud.aiplatform.compat.types import (
    matching_engine_index as gca_matching_engine_index,
)

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
    "index_endpoint_name",
    "deployed_index_id",
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
        - index_endpoint_name: Matching Engine index endpoint resource name or ID.
        - deployed_index_id: ID of the deployed index on the endpoint.
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
        index_endpoint_name: Optional[str] = None,
        deployed_index_id: Optional[str] = None,
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
            index_endpoint_name: Index endpoint resource name or ID.
            deployed_index_id: Deployed index ID within the endpoint.
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
        self.index_endpoint_name: Optional[str] = params.get(
            "index_endpoint_name"
        )
        self.deployed_index_id: Optional[str] = params.get(
            "deployed_index_id"
        )
        self.credentials: Optional[str] = params.get(
            "credentials"
        )
        self.api_key: Optional[str] = params.get("api_key")
        self.timeout: Optional[int] = params.get("timeout")
        self._initialized: bool = False
        self._index: Optional[aiplatform.MatchingEngineIndex] = None
        self._endpoint: Optional[aiplatform.MatchingEngineIndexEndpoint] = None

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
                self._index = aiplatform.MatchingEngineIndex(
                    index_name=self.index_name,
                    project=self.project_id,
                    location=self.region,
                )
            if self.index_endpoint_name:
                self._endpoint = aiplatform.MatchingEngineIndexEndpoint(
                    index_endpoint_name=self.index_endpoint_name,
                    project=self.project_id,
                    location=self.region,
                )
            self.client = self._index or self._endpoint
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
            if self.index_name and self._index is None:
                self._index = aiplatform.MatchingEngineIndex(
                    index_name=self.index_name,
                    project=self.project_id,
                    location=self.region,
                )
            if self.index_endpoint_name and self._endpoint is None:
                self._endpoint = aiplatform.MatchingEngineIndexEndpoint(
                    index_endpoint_name=self.index_endpoint_name,
                    project=self.project_id,
                    location=self.region,
                )
            return bool(self._index or self._endpoint)
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
        return sorted({idx.display_name for idx in indexes})

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
        """Upsert datapoints into a Matching Engine index.

        Note: Vertex Matching Engine does not store arbitrary payload
        metadata alongside datapoints in the same way as other vector
        databases. This adapter upserts only the datapoint ID and
        feature vector.
        """
        if not self._initialized:
            self._init_client_context()
        if self._index is None:
            raise RuntimeError(
                "Vertex AI index client is not initialized. "
                "Provide 'index_name' and call connect()."
            )
        datapoints = [
            gca_matching_engine_index.IndexDatapoint(
                datapoint_id=record.id,
                feature_vector=record.vector,
            )
            for record in records
        ]
        self._index.upsert_datapoints(datapoints=datapoints, **kwargs)

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 5,
        **kwargs: Any,
    ) -> list[VectorSearchResult]:
        """Run nearest-neighbor search via Matching Engine index endpoint."""
        if not self._initialized:
            self._init_client_context()
        if self._endpoint is None:
            raise RuntimeError(
                "Vertex AI index endpoint client is not initialized. "
                "Provide 'index_endpoint_name' and call connect()."
            )
        if not self.deployed_index_id:
            raise RuntimeError(
                "Missing 'deployed_index_id' for Vertex AI neighbor search."
            )
        neighbors = self._endpoint.find_neighbors(
            deployed_index_id=self.deployed_index_id,
            queries=[query_vector],
            num_neighbors=limit,
            **kwargs,
        )
        first = neighbors[0] if neighbors else []
        results: list[VectorSearchResult] = []
        for n in first:
            results.append(
                VectorSearchResult(
                    id=str(getattr(n, "id", "")),
                    score=float(getattr(n, "distance", 0.0) or 0.0),
                    payload={},
                )
            )
        return results

    def delete(
        self,
        collection_name: str,
        ids: list[str],
        **kwargs: Any,
    ) -> None:
        """Remove datapoints from a Matching Engine index."""
        if not self._initialized:
            self._init_client_context()
        if self._index is None:
            raise RuntimeError(
                "Vertex AI index client is not initialized. "
                "Provide 'index_name' and call connect()."
            )
        self._index.remove_datapoints(datapoint_ids=ids, **kwargs)
{%- endif -%}
