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
from .protocols import CollectionConfig, VectorDB, VectorDBConfig
from .types import (
    CollectionConfig as CollectionConfigDTO,
    DistanceMetric,
    VectorDBConfig as VectorDBConfigDTO,
    VectorDBProvider,
)

__all__: list[str] = [
    # Protocols
    "VectorDBConfig",
    "CollectionConfig",
    "VectorDB",
    # Concrete types
    "VectorDBConfigDTO",
    "CollectionConfigDTO",
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
