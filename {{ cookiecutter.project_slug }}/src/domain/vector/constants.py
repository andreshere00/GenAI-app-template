{%- if cookiecutter.vector_db -%}
{% if "1" in cookiecutter.vector_db %}
COSMOS_PARAM_MAP: dict[str, str] = {
    "api_key": "credential",
    "timeout": "connection_timeout",
}
{% endif %}
{%- if "2" in cookiecutter.vector_db %}
MILVUS_PARAM_MAP: dict[str, str] = {
    "url": "uri",
    "api_key": "token",
    "database": "db_name",
}
{% endif %}
{%- if "3" in cookiecutter.vector_db %}
MONGO_PARAM_MAP: dict[str, str] = {
    "timeout": "serverSelectionTimeoutMS",
}
{% endif %}
{%- if "4" in cookiecutter.vector_db %}
OPENSEARCH_PARAM_MAP: dict[str, str] = {}
{% endif %}
{%- if "5" in cookiecutter.vector_db %}
PINECONE_PARAM_MAP: dict[str, str] = {
    "collection": "index_name",
}
{% endif %}
{%- if "6" in cookiecutter.vector_db %}
QDRANT_PARAM_MAP: dict[str, str] = {}
{% endif %}
{%- if "7" in cookiecutter.vector_db %}
VERTEX_PARAM_MAP: dict[str, str] = {
    "database": "project_id",
    "collection": "index_name",
}
{% endif %}
{%- endif -%}
