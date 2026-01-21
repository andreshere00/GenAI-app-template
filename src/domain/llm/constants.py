# LLMs

## OpenAI

OPENAI_PARAM_MAP: dict[str, str] = {
    "api_key": "openai_api_key",
    "base_url": "openai_api_base",
    "proxy": "openai_proxy",
    "organization": "openai_organization",
}

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

## Gemini

GEMINI_PARAM_MAP: dict[str, str] = {
    "api_key": "google_api_key",
    "timeout": "request_timeout",
}

## Bedrock

BEDROCK_PARAM_MAP: dict[str, str] = {
    "model": "model_id",
    "base_url": "endpoint_url",
    "timeout": "client_config",
}

## Grok

XAI_PARAM_MAP: dict[str, str] = {
    "api_key": "xai_api_key",
    "base_url": "xai_api_base",
}

## Anthropic

CLAUDE_PARAM_MAP: dict[str, str] = {
    "timeout": "default_request_timeout",
    "proxy": "anthropic_proxy",
    "base_url": "anthropic_api_url",
    "api_key": "api_key",
}
