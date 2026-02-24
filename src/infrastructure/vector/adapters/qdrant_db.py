from typing import Any, Optional

from qdrant_client import QdrantClient
from ....domain.vector import VectorDBConfig
from ...utils import resolve_parameters


class QdrantVectorDatabase:

    def __init__(
        self, 
        config: Optional[VectorDBConfig] = None,
        *,
        host: Optional[str] = None,
        port: Optional[int] = None,
        **kwargs: Any,
        ) -> None:

        params = resolve_parameters(config, host=host, port=port, **kwargs)
        self.client = QdrantClient(**params)