# LLMs
{% if "1" in cookiecutter.llm_providers %}
## Anthropic

CLAUDE_PARAM_MAP: dict[str, str] = {
    "timeout": "default_request_timeout",
    "proxy": "anthropic_proxy",
    "base_url": "anthropic_api_url",
    "api_key": "api_key",
}

{%- endif -%}
{% if "2" in cookiecutter.llm_providers %}

## Azure

AZURE_OPENAI_PARAM_MAP: dict[str, str] = {
    # Azure specific mappings
    "base_url": "azure_endpoint",
    "model": "azure_deployment",
    "api_version": "api_version",
    "api_key": "api_key",
    # Standard mappings
    "proxy": "openai_proxy",
    "organization": "openai_organization",
}

{%- endif -%}
{% if "3" in cookiecutter.llm_providers %}

## Bedrock

BEDROCK_PARAM_MAP: dict[str, str] = {
    "model": "model_id",
    "base_url": "endpoint_url",
    "timeout": "client_config",
}

{%- endif -%}
{% if "4" in cookiecutter.llm_providers %}

## Gemini

GEMINI_PARAM_MAP: dict[str, str] = {
    "api_key": "google_api_key",
    "timeout": "request_timeout",
}

{%- endif -%}
{% if "5" in cookiecutter.llm_providers %}

## Grok

XAI_PARAM_MAP: dict[str, str] = {
    "api_key": "xai_api_key",
    "base_url": "xai_api_base",
}

{%- endif -%}
{% if "7" in cookiecutter.llm_providers %}

## OpenAI

OPENAI_PARAM_MAP: dict[str, str] = {
    "api_key": "openai_api_key",
    "base_url": "openai_api_base",
    "proxy": "openai_proxy",
    "organization": "openai_organization",
}

{%- endif -%}