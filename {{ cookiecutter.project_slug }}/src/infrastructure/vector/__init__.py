{%- if cookiecutter.vector_db -%}
"""Infrastructure layer for vector database integrations."""

from .base import BaseVectorDatabase
{% if "1" in cookiecutter.vector_db %}
from .adapters import CosmosDBVectorDatabase
{%- endif %}
{% if "2" in cookiecutter.vector_db %}
from .adapters import MilvusVectorDatabase
{%- endif %}
{% if "3" in cookiecutter.vector_db %}
from .adapters import MongoDBVectorDatabase
{%- endif %}
{% if "4" in cookiecutter.vector_db %}
from .adapters import OpenSearchVectorDatabase
{%- endif %}
{% if "5" in cookiecutter.vector_db %}
from .adapters import PineconeVectorDatabase
{%- endif %}
{% if "6" in cookiecutter.vector_db %}
from .adapters import QdrantVectorDatabase
{%- endif %}
{% if "7" in cookiecutter.vector_db %}
from .adapters import VertexDBVectorDatabase
{%- endif %}

__all__ = [
    "BaseVectorDatabase",
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
