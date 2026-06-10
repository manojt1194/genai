from pydantic import BaseModel, Field


class Message(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)


class ChatRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    message: str = Field(..., min_length=1, max_length=1000)


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=5000)
    
class MemoryChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=100)
    message: str = Field(..., min_length=1, max_length=1000)