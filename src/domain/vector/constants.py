MILVUS_PARAM_MAP: dict[str, str] = {
    "url": "uri",
    "api_key": "token",
    "database": "db_name",
}

QDRANT_PARAM_MAP: dict[str, str] = {}

PINECONE_PARAM_MAP: dict[str, str] = {
    "collection": "index_name",
}

COSMOS_PARAM_MAP: dict[str, str] = {
    "api_key": "credential",
    "timeout": "connection_timeout",
}

OPENSEARCH_PARAM_MAP: dict[str, str] = {}

MONGO_PARAM_MAP: dict[str, str] = {
    "timeout": "serverSelectionTimeoutMS",
}

VERTEX_PARAM_MAP: dict[str, str] = {
    "database": "project_id",
    "collection": "index_name",
}

AZURE_AI_SEARCH_PARAM_MAP: dict[str, str] = {
    "url": "endpoint",
    "api_key": "credential",
}
