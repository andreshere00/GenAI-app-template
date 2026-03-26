{%- if cookiecutter.vector_db -%}
"""Domain models for vector databases."""

from .types import VectorDBConfig

__all__ = ["VectorDBConfig"]
{%- endif -%}
