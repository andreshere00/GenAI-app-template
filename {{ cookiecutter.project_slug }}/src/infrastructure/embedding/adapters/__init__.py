{%- if "1" in cookiecutter.embedding_providers -%}
from .azure_openai import AzureOpenAIEmbeddingModel
{% endif %}
{% if "2" in cookiecutter.embedding_providers %}
from .bedrock import BedrockEmbeddingModel
{% endif %}
{% if "3" in cookiecutter.embedding_providers %}
from .cohere import CohereEmbeddingModel
{% endif %}
{% if "4" in cookiecutter.embedding_providers %}
from .gemini import GeminiEmbeddingModel
{% endif %}
{% if "5" in cookiecutter.embedding_providers %}
from .grok import GrokEmbeddingModel
{% endif %}
{% if "6" in cookiecutter.embedding_providers %}
from .ollama import OllamaEmbeddingModel
{% endif %}
{%- if "7" in cookiecutter.embedding_providers -%}
from .openai import OpenAIEmbeddingModel
{% endif %}
{% if "8" in cookiecutter.embedding_providers %}
from .voyageai import VoyageAIEmbeddingModel
{% endif %}


{%- if cookiecutter.embedding_providers -%}__all__ = [
    {%- if "1" in cookiecutter.embedding_providers -%}
    "AzureOpenAIEmbeddingModel",
    {%- endif -%}
    {%- if "2" in cookiecutter.embedding_providers -%}
    "BedrockEmbeddingModel",
    {%- endif -%}
    {%- if "3" in cookiecutter.embedding_providers -%}
    "CohereEmbeddingModel",
    {%- endif -%}
    {%- if "4" in cookiecutter.embedding_providers -%}
    "GeminiEmbeddingModel",
    {%- endif -%}
    {%- if "5" in cookiecutter.embedding_providers -%}
    "GrokEmbeddingModel",
    {%- endif -%}
    {%- if "6" in cookiecutter.embedding_providers -%}
    "OllamaEmbeddingModel",
    {%- endif -%}
    {%- if "7" in cookiecutter.embedding_providers -%}
    "OpenAIEmbeddingModel",
    {%- endif -%}
    {%- if "8" in cookiecutter.embedding_providers -%}
    "VoyageAIEmbeddingModel",
    {%- endif -%}
]{%- endif -%}
