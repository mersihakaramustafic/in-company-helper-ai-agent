from typing import Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str


class FeedbackRequest(BaseModel):
    trace_id: str
    helpful: bool
    comment: Optional[str] = None
