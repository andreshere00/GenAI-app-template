from enum import Enum
from typing import Any, List, Optional, Protocol
from pydantic import BaseModel, Field


class ChatMode(Enum):
    DIRECT = "direct"
    STREAM = "stream"
    BATCH = "batch"


class ChatMessage(BaseModel):
    """Represents a message in a chat conversation."""

    role: str
    content: str
    metadata: Optional[dict[str, Any]] = Field(default_factory=dict)
