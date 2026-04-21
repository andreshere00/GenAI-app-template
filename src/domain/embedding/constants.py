# Embeddings

## Azure OpenAI

AZURE_OPENAI_EMBEDDING_PARAM_MAP: dict[str, str] = {
    "base_url": "azure_endpoint",
    "model": "azure_deployment",
    "api_version": "openai_api_version",
    "api_key": "api_key",
}

## Bedrock

BEDROCK_EMBEDDING_PARAM_MAP: dict[str, str] = {
    "model": "model_id",
    "base_url": "endpoint_url",
}

## Cohere

COHERE_EMBEDDING_PARAM_MAP: dict[str, str] = {
    "api_key": "cohere_api_key",
}

## Gemini

GEMINI_EMBEDDING_PARAM_MAP: dict[str, str] = {
    "api_key": "google_api_key",
}

## Grok (xAI – OpenAI-compatible)

XAI_EMBEDDING_PARAM_MAP: dict[str, str] = {
    "api_key": "openai_api_key",
    "base_url": "openai_api_base",
}

## OpenAI

OPENAI_EMBEDDING_PARAM_MAP: dict[str, str] = {
    "api_key": "openai_api_key",
    "base_url": "openai_api_base",
    "organization": "openai_organization",
}

## VoyageAI

VOYAGEAI_EMBEDDING_PARAM_MAP: dict[str, str] = {
    "api_key": "voyage_api_key",
}
