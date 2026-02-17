from typing import Any, Dict, Optional, Protocol, Union

from pydantic import SecretStr


class ModelConfig(Protocol):
    """Protocol defining the configuration shape for LLM models (Standard & Azure).

    This protocol is extensible. Generic fields like 'model' or 'base_url' can
    be mapped to specific implementation arguments (e.g., 'azure_deployment')
    via the adapter classes.
    """

    api_key: Union[str, SecretStr]
    model: Optional[str]

    # Standard Connection
    base_url: Optional[str]
    organization: Optional[str]

    # Azure Specifics
    azure_endpoint: Optional[str]
    azure_deployment: Optional[str]
    api_version: Optional[str]

    # Common Configuration
    timeout: Optional[Union[float, int]]
    max_retries: Optional[int]
    proxy: Optional[str]
    http_client: Optional[Any]

    # Generation parameters
    temperature: Optional[float]
    top_p: Optional[float]
    model_kwargs: Optional[Dict[str, Any]]
