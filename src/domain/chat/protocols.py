from typing import Protocol, Optional
from .types import ChatMode


class ChatConfig(Protocol):
    """Protocol defining chat configuration parameters."""

    mode: ChatMode
    max_history: int
    temperature: Optional[float]
