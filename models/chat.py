from dataclasses import dataclass, field
from typing import Optional, Literal
from datetime import datetime, UTC


@dataclass
class ChatMessage:
    session: str
    message: str
    sender: Literal["user", "assistant", "system"] = "user"
    message_id: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
