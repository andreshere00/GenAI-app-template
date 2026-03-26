from dataclasses import asdict, is_dataclass
from typing import Any, Optional


def _config_to_dict(config: Optional[Any]) -> dict[str, Any]:
    """Convert supported config shapes into a dictionary."""
    if config is None:
        return {}

    if isinstance(config, dict):
        return dict(config)

    if is_dataclass(config):
        return asdict(config)

    # Pydantic v2
    if hasattr(config, "model_dump") and callable(config.model_dump):
        return config.model_dump()

    # Pydantic v1
    if hasattr(config, "dict") and callable(config.dict):
        return config.dict()

    return vars(config)


def resolve_parameters(
    config: Optional[Any],
    *,
    allowed_keys: Optional[set[str]] = None,
    aliases: Optional[dict[str, str]] = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Merges configuration object attributes with explicit overrides.

    Args:
        config: The source configuration object.
        **overrides: Dictionary of arguments passed directly to __init__.

    Returns:
        A dictionary containing the final non-None parameters.
    """
    final_params: dict[str, Any] = {}
    aliases = aliases or {}

    source = _config_to_dict(config)
    source_kwargs = source.pop("kwargs", None)
    if isinstance(source_kwargs, dict):
        source.update(source_kwargs)

    for key, value in source.items():
        if value is None:
            continue
        mapped_key = aliases.get(key, key)
        if allowed_keys is None or mapped_key in allowed_keys:
            final_params[mapped_key] = value

    for key, value in overrides.items():
        if value is None:
            continue
        mapped_key = aliases.get(key, key)
        if allowed_keys is None or mapped_key in allowed_keys:
            final_params[mapped_key] = value

    return final_params