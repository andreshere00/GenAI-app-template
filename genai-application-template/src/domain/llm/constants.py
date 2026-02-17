# LLMs

## Anthropic

CLAUDE_PARAM_MAP: dict[str, str] = {
    "timeout": "default_request_timeout",
    "proxy": "anthropic_proxy",
    "base_url": "anthropic_api_url",
    "api_key": "api_key",
}

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

## Grok

XAI_PARAM_MAP: dict[str, str] = {
    "api_key": "xai_api_key",
    "base_url": "xai_api_base",
}