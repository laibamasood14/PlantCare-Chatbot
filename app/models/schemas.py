from typing import List, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User's plant care question")
    chat_history: Optional[List[ChatMessage]] = Field(default=[], description="Previous turns")


class SourceInfo(BaseModel):
    source: str
    page: int


class ChatResponse(BaseModel):
    answer: str
    language_detected: str = Field(..., description="'en' or 'ur'")
    sources: List[SourceInfo]
    query: str

