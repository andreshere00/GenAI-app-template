# Embeddings
{% if "1" in cookiecutter.embedding_providers %}
## Azure OpenAI

AZURE_OPENAI_EMBEDDING_PARAM_MAP: dict[str, str] = {
    "base_url": "azure_endpoint",
    "model": "azure_deployment",
    "api_version": "openai_api_version",
    "api_key": "api_key",
}

{%- endif -%}
{% if "2" in cookiecutter.embedding_providers %}

## Bedrock

BEDROCK_EMBEDDING_PARAM_MAP: dict[str, str] = {
    "model": "model_id",
    "base_url": "endpoint_url",
}

{%- endif -%}
{% if "3" in cookiecutter.embedding_providers %}

## Cohere

COHERE_EMBEDDING_PARAM_MAP: dict[str, str] = {
    "api_key": "cohere_api_key",
}

{%- endif -%}
{% if "4" in cookiecutter.embedding_providers %}

## Gemini

GEMINI_EMBEDDING_PARAM_MAP: dict[str, str] = {
    "api_key": "google_api_key",
}

{%- endif -%}
{% if "5" in cookiecutter.embedding_providers %}

## Grok (xAI – OpenAI-compatible)

XAI_EMBEDDING_PARAM_MAP: dict[str, str] = {
    "api_key": "openai_api_key",
    "base_url": "openai_api_base",
}

{%- endif -%}
{% if "7" in cookiecutter.embedding_providers %}

## OpenAI

OPENAI_EMBEDDING_PARAM_MAP: dict[str, str] = {
    "api_key": "openai_api_key",
    "base_url": "openai_api_base",
    "organization": "openai_organization",
}

{%- endif -%}
{% if "8" in cookiecutter.embedding_providers %}

## VoyageAI

VOYAGEAI_EMBEDDING_PARAM_MAP: dict[str, str] = {
    "api_key": "voyage_api_key",
}

{%- endif -%}
