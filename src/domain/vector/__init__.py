"""Domain models for vector databases."""

from .constants import (
    AZURE_AI_SEARCH_PARAM_MAP,
    COSMOS_PARAM_MAP,
    MILVUS_PARAM_MAP,
    MONGO_PARAM_MAP,
    OPENSEARCH_PARAM_MAP,
    PINECONE_PARAM_MAP,
    QDRANT_PARAM_MAP,
    VERTEX_PARAM_MAP,
)
from .protocols import (
    CollectionConfig,
    VectorDB,
    VectorDBConfig,
    VectorRecord,
    VectorSearchResult,
)
from .types import (
    CollectionConfig as CollectionConfigDTO,
    DistanceMetric,
    VectorDBConfig as VectorDBConfigDTO,
    VectorRecord as VectorRecordDTO,
    VectorSearchResult as VectorSearchResultDTO,
    VectorDBProvider,
)

__all__: list[str] = [
    # Protocols
    "VectorDBConfig",
    "CollectionConfig",
    "VectorDB",
    "VectorRecord",
    "VectorSearchResult",
    # Concrete types
    "VectorDBConfigDTO",
    "CollectionConfigDTO",
    "VectorRecordDTO",
    "VectorSearchResultDTO",
    "VectorDBProvider",
    "DistanceMetric",
    # Parameter maps
    "MILVUS_PARAM_MAP",
    "QDRANT_PARAM_MAP",
    "PINECONE_PARAM_MAP",
    "COSMOS_PARAM_MAP",
    "OPENSEARCH_PARAM_MAP",
    "MONGO_PARAM_MAP",
    "VERTEX_PARAM_MAP",
    "AZURE_AI_SEARCH_PARAM_MAP",
]
