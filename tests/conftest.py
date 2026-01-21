from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pytest


@dataclass
class MockConfig:
    """Mock implementation of ModelConfig protocol for testing purposes."""

    api_key: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    base_url: Optional[str] = None
    organization: Optional[str] = None
    timeout: Optional[float] = None
    max_retries: Optional[int] = None
    proxy: Optional[str] = None
    http_client: Optional[Any] = None
    top_p: Optional[float] = None
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    azure_endpoint: Optional[str] = None
    azure_deployment: Optional[str] = None
    api_version: Optional[str] = None


@pytest.fixture
def config_factory() -> Callable[..., MockConfig]:
    """Fixture that returns a factory function to create MockConfig instances.

    Returns:
        Callable: A function that accepts kwargs and returns a MockConfig.
    """

    def _create_config(**kwargs: Any) -> MockConfig:
        return MockConfig(**kwargs)

    return _create_config
