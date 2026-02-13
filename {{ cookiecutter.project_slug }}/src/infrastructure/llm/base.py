from typing import Any, Dict, Optional

from ...domain.llm.protocols import ModelConfig


class BaseLlm:
    """Base class for all LLM wrappers implementing common configuration logic.

    This class provides the mechanisms to merge configuration objects with explicit
    overrides and map generic parameter names to provider-specific arguments.
    """

    _PARAM_MAP: Dict[str, str] = {}

    def _resolve_parameters(
        self, config: Optional[ModelConfig], **overrides: Any
    ) -> Dict[str, Any]:
        """Merge configuration object attributes with explicit overrides.

        Args:
            config: The source configuration object.
            **overrides: Dictionary of arguments passed directly to __init__.

        Returns:
            A dictionary containing the final non-None parameters for instantiation.
        """
        final_params: Dict[str, Any] = {}

        if config:
            # Safely extract known fields from the Protocol
            protocol_fields = ModelConfig.__annotations__.keys()
            for field in protocol_fields:
                if hasattr(config, field):
                    value = getattr(config, field)
                    if value is not None:
                        final_params[field] = value

        for key, value in overrides.items():
            if value is not None:
                final_params[key] = value

        return final_params

    def _map_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the provider-specific parameter mapping.

        This method mutates the params dictionary by renaming keys according to
        the class's _PARAM_MAP.

        Args:
            params: The dictionary of resolved parameters.

        Returns:
            The parameters dictionary with keys mapped to the specific client args.
        """
        for config_key, langchain_key in self._PARAM_MAP.items():
            if config_key in params:
                # Only map if the target key is not already explicitly provided
                if langchain_key not in params:
                    params[langchain_key] = params.pop(config_key)
        return params
