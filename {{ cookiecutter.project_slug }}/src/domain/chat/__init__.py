{%- if "chat" in cookiecutter.services -%}
from .types import ChatMessage, ChatMode
from .protocols import ChatConfig

__all__: list[str] = [
    "ChatMessage",
    "ChatMode",
    "ChatConfig",
]
{%- endif -%}
