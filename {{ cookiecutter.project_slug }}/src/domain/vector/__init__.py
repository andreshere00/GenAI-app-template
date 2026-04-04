{%- if cookiecutter.vector_db -%}
"""Domain models for vector databases."""

from .protocols import CollectionConfig, VectorDB, VectorDBConfig
from .types import (
    CollectionConfig as CollectionConfigDTO,
    DistanceMetric,
    VectorDBConfig as VectorDBConfigDTO,
    VectorDBProvider,
)
{% if "1" in cookiecutter.vector_db %}
from .constants import COSMOS_PARAM_MAP
{%- endif %}
{% if "2" in cookiecutter.vector_db %}
from .constants import MILVUS_PARAM_MAP
{%- endif %}
{% if "3" in cookiecutter.vector_db %}
from .constants import MONGO_PARAM_MAP
{%- endif %}
{% if "4" in cookiecutter.vector_db %}
from .constants import OPENSEARCH_PARAM_MAP
{%- endif %}
{% if "5" in cookiecutter.vector_db %}
from .constants import PINECONE_PARAM_MAP
{%- endif %}
{% if "6" in cookiecutter.vector_db %}
from .constants import QDRANT_PARAM_MAP
{%- endif %}
{% if "7" in cookiecutter.vector_db %}
from .constants import VERTEX_PARAM_MAP
{%- endif %}

__all__: list[str] = [
    "VectorDBConfig",
    "CollectionConfig",
    "VectorDB",
    "VectorDBConfigDTO",
    "CollectionConfigDTO",
    "VectorDBProvider",
    "DistanceMetric",
    {%- if "1" in cookiecutter.vector_db %}
    "COSMOS_PARAM_MAP",
    {%- endif %}
    {%- if "2" in cookiecutter.vector_db %}
    "MILVUS_PARAM_MAP",
    {%- endif %}
    {%- if "3" in cookiecutter.vector_db %}
    "MONGO_PARAM_MAP",
    {%- endif %}
    {%- if "4" in cookiecutter.vector_db %}
    "OPENSEARCH_PARAM_MAP",
    {%- endif %}
    {%- if "5" in cookiecutter.vector_db %}
    "PINECONE_PARAM_MAP",
    {%- endif %}
    {%- if "6" in cookiecutter.vector_db %}
    "QDRANT_PARAM_MAP",
    {%- endif %}
    {%- if "7" in cookiecutter.vector_db %}
    "VERTEX_PARAM_MAP",
    {%- endif %}
]
{%- endif -%}
