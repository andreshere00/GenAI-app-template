{%- if cookiecutter.vector_db -%}
"""Vector database adapters for various providers."""
{% if "1" in cookiecutter.vector_db %}
from .cosmos_db import CosmosDBVectorDatabase
{%- endif %}
{% if "2" in cookiecutter.vector_db %}
from .milvus_db import MilvusVectorDatabase
{%- endif %}
{% if "3" in cookiecutter.vector_db %}
from .mongo_db import MongoDBVectorDatabase
{%- endif %}
{% if "4" in cookiecutter.vector_db %}
from .opensearch_db import OpenSearchVectorDatabase
{%- endif %}
{% if "5" in cookiecutter.vector_db %}
from .pinecone_db import PineconeVectorDatabase
{%- endif %}
{% if "6" in cookiecutter.vector_db %}
from .qdrant_db import QdrantVectorDatabase
{%- endif %}
{% if "7" in cookiecutter.vector_db %}
from .vertex_db import VertexDBVectorDatabase
{%- endif %}

__all__ = [
    {%- if "1" in cookiecutter.vector_db %}
    "CosmosDBVectorDatabase",
    {%- endif %}
    {%- if "2" in cookiecutter.vector_db %}
    "MilvusVectorDatabase",
    {%- endif %}
    {%- if "3" in cookiecutter.vector_db %}
    "MongoDBVectorDatabase",
    {%- endif %}
    {%- if "4" in cookiecutter.vector_db %}
    "OpenSearchVectorDatabase",
    {%- endif %}
    {%- if "5" in cookiecutter.vector_db %}
    "PineconeVectorDatabase",
    {%- endif %}
    {%- if "6" in cookiecutter.vector_db %}
    "QdrantVectorDatabase",
    {%- endif %}
    {%- if "7" in cookiecutter.vector_db %}
    "VertexDBVectorDatabase",
    {%- endif %}
]
{%- endif -%}
