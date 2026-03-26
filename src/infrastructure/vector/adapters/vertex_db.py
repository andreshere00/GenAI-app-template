from typing import Any, Optional

from google.cloud import aiplatform

from ....domain.vector import VectorDBConfig
from ...utils import resolve_parameters
from ..base import BaseVectorDatabase

VERTEX_ALLOWED_KEYS = {
    "project_id",
    "region",
    "index_name",
    "credentials",
    "api_key",
    "timeout",
}


class VertexDBVectorDatabase(BaseVectorDatabase):
    """Wrapper for Google Vertex AI Vector Search client.

    Google Vertex AI Vector Search (formerly Matching Engine) is a managed
    vector database service on Google Cloud Platform. This adapter manages
    connections to Vertex AI Vector Search indexes.

    Configuration parameters:
        - project_id: Google Cloud Project ID (required)
        - region: GCP region for the service (default: "us-central1")
        - index_name: Name of the vector search index
        - credentials: Path to service account JSON key (optional)
        - api_key: Google Cloud API key (alternative to credentials)
        - timeout: Request timeout in seconds
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
            credentials: Path to service account credentials JSON file.
            api_key: Google API key for authentication.
            timeout: Request timeout in seconds.
            **kwargs: Additional Vertex AI-specific parameters.
        """
        super().__init__(config, **kwargs)

        params = resolve_parameters(
            config,
            allowed_keys=VERTEX_ALLOWED_KEYS,
            aliases={"database": "project_id", "collection": "index_name"},
            project_id=project_id,
            region=region,
            index_name=index_name,
            credentials=credentials,
            api_key=api_key,
            timeout=timeout,
            **kwargs,
        )

        self.project_id = params.get("project_id")
        self.region = params.get("region") or "us-central1"
        self.index_name = params.get("index_name")
        self.credentials = params.get("credentials")
        self.api_key = params.get("api_key")
        self.timeout = params.get("timeout")
        self._initialized = False

        # Note: Actual index client will be created on-demand
        # The Vertex AI SDK doesn't require explicit connection
        self.client = None

    def _init_client_context(self) -> None:
        """Initialize SDK context for this adapter instance."""
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

    def connect(self) -> None:
        """Establish connection to Vertex AI Vector Search.

        Vertex AI uses automatic authentication via Application Default
        Credentials (ADC) or service account key.
        """
        try:
            if not self._initialized:
                self._init_client_context()
            if self.index_name:
                # Attempt to retrieve index to verify connectivity
                index = aiplatform.MatchingEngineIndex(
                    index_name=self.index_name,
                    project=self.project_id,
                    location=self.region,
                )
                self.client = index
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to Vertex AI index {self.index_name}: {e}"
            )

    def disconnect(self) -> None:
        """Close the Vertex AI Vector Search connection.

        Vertex AI automatically manages connections, so no cleanup needed.
        """

    def health(self) -> bool:
        """Check if Vertex AI Vector Search connection is healthy.

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
