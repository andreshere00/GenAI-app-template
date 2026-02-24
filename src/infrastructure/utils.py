from typing import Any, Optional


def resolve_parameters(config: Optional[Any], **overrides: Any) -> dict[str, Any]:
    """Merges configuration object attributes with explicit overrides.

    Args:
        config: The source configuration object.
        **overrides: Dictionary of arguments passed directly to __init__.

    Returns:
        A dictionary containing the final non-None parameters.
    """
    final_params: dict[str, Any] = {}

    if config:
        # Assuming config is a Protocol or has __annotations__
        protocol_fields = getattr(config, '__annotations__', {}).keys()
        for field in protocol_fields:
            if hasattr(config, field):
                value = getattr(config, field)
                if value is not None:
                    final_params[field] = value

    for key, value in overrides.items():
        if value is not None:
            final_params[key] = value

    return final_params