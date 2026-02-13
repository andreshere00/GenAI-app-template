from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

from pydantic import SecretStr

from src.domain.llm.protocols import ModelConfig
from src.infrastructure.llm.base import BaseLlm

# --- Fixtures & Mocks ---


@dataclass
class MockConfig:
    """A simple dataclass that implements the ModelConfig protocol."""

    api_key: Union[str, SecretStr]
    model: Optional[str] = None
    base_url: Optional[str] = None
    timeout: Optional[float] = None
    temperature: Optional[float] = None
    # Add other fields required by protocol if strictly checked,
    # but for Duck Typing this is usually sufficient for these tests.
    organization: Optional[str] = None
    azure_endpoint: Optional[str] = None
    azure_deployment: Optional[str] = None
    api_version: Optional[str] = None
    max_retries: Optional[int] = None
    proxy: Optional[str] = None
    http_client: Optional[Any] = None
    top_p: Optional[float] = None
    model_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)


class ConcreteTestLLM(BaseLlm):
    """A concrete implementation of BaseLlm for testing purposes.

    We simulate a provider that requires 'test_api_key' instead of 'api_key',
    and 'mapped_model' instead of 'model'.
    """

    _PARAM_MAP = {
        "api_key": "test_api_key",
        "model": "mapped_model",
        "timeout": "client_timeout",
    }

    def __init__(self, config: Optional[ModelConfig] = None, **kwargs):
        # We expose the internal params processing to verify the logic
        self.resolved_params = self._resolve_parameters(config, **kwargs)
        self.final_params = self._map_parameters(self.resolved_params.copy())


# --- Test Suite ---


class TestBaseLlm:
    def test_resolve_parameters_config_only(self):
        """Test that parameters are correctly extracted from the config object."""
        config = MockConfig(api_key="secret", model="gpt-test", temperature=0.7)
        llm = ConcreteTestLLM(config=config)

        assert llm.resolved_params["api_key"] == "secret"
        assert llm.resolved_params["model"] == "gpt-test"
        assert llm.resolved_params["temperature"] == 0.7
        # Ensure None values are not included
        assert "base_url" not in llm.resolved_params

    def test_resolve_parameters_kwargs_only(self):
        """Test that parameters are correctly extracted from kwargs."""
        llm = ConcreteTestLLM(api_key="kwarg-key", temperature=0.9)

        assert llm.resolved_params["api_key"] == "kwarg-key"
        assert llm.resolved_params["temperature"] == 0.9

    def test_resolve_parameters_override_precedence(self):
        """Test that kwargs override config values."""
        config = MockConfig(api_key="config-key", temperature=0.5)

        # Override temperature, keep api_key from config
        llm = ConcreteTestLLM(config=config, temperature=1.0)

        assert llm.resolved_params["api_key"] == "config-key"
        assert llm.resolved_params["temperature"] == 1.0

    def test_map_parameters_renaming(self):
        """Test that keys are correctly renamed based on _PARAM_MAP."""
        # Config has standard keys: 'api_key', 'model'
        config = MockConfig(api_key="secret", model="v1")
        llm = ConcreteTestLLM(config=config)

        # internal logic should have renamed them
        assert "api_key" not in llm.final_params
        assert "model" not in llm.final_params

        assert llm.final_params["test_api_key"] == "secret"
        assert llm.final_params["mapped_model"] == "v1"

    def test_map_parameters_no_rename_if_missing(self):
        """Test that mapping logic doesn't crash if source keys are missing."""
        # 'timeout' is in the map but not in config/kwargs
        config = MockConfig(api_key="secret")
        llm = ConcreteTestLLM(config=config)

        assert "client_timeout" not in llm.final_params

    def test_map_parameters_skip_if_target_exists(self):
        """Test that mapping does not overwrite if the target key is already provided.

        Scenario: User knows the internal parameter name (e.g., 'mapped_model')
        and passes it directly via kwargs. We should use that instead of mapping
        'model' from config.
        """
        config = MockConfig(api_key="secret", model="config_model")

        # User explicitly passes 'mapped_model' which is the target key
        llm = ConcreteTestLLM(config=config, mapped_model="explicit_override")

        # Expect 'mapped_model' to be 'explicit_override', NOT 'config_model'
        assert llm.final_params["mapped_model"] == "explicit_override"

        # The original 'model' key might remain or be popped depending on implementation.
        # Your current logic: params[langchain_key] = params.pop(config_key)
        # BUT only 'if langchain_key not in params'.
        # So 'model' (config_key) should STILL exist in params because we didn't pop it.
        assert "model" in llm.final_params
        assert llm.final_params["model"] == "config_model"

    def test_secret_str_handling(self):
        """Test that SecretStr is accepted (pydantic compat)."""
        secret = SecretStr("secure-token")
        config = MockConfig(api_key=secret)
        llm = ConcreteTestLLM(config=config)

        assert llm.resolved_params["api_key"] == secret
        assert llm.final_params["test_api_key"] == secret

    def test_unmapped_parameters_preserved(self):
        """Test that parameters not in the map are preserved as-is."""
        llm = ConcreteTestLLM(api_key="x", random_param="preserved")

        assert llm.final_params["random_param"] == "preserved"
