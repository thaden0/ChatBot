from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import datetime, UTC


class ChatMessageCreate(BaseModel):
    session: str = Field(..., description="Conversation/session ID")
    message: str = Field(..., description="Message content")


class ChatMessageResponse(BaseModel):
    session: str
    message: str
    sender: Literal["user", "assistant", "system"] = "user"
    message_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
