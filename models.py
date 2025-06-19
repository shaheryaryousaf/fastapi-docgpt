from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime
from enum import Enum

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500, description="The question to ask")
    
    @validator('question')
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError('Question cannot be empty or just whitespace')
        return v.strip()

class QuestionResponse(BaseModel):
    answer: str
    confidence_score: Optional[float] = None
    sources: Optional[List[str]] = None
    processing_time: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class UploadResponse(BaseModel):
    message: str
    document_id: Optional[str] = None
    chunks_processed: Optional[int] = None
    processing_time: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ErrorResponse(BaseModel):
    error: str
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    dependencies: Optional[dict] = None

class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    file_size: int
    upload_time: datetime
    status: ProcessingStatus
    chunks_count: Optional[int] = None
    error_message: Optional[str] = None 