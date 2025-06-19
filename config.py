from pydantic_settings import BaseSettings
from typing import List, Optional
import os

class Settings(BaseSettings):
    # API Configuration
    app_name: str = "FastAPI DocGPT"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    
    # OpenAI Configuration
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-ada-002"
    openai_temperature: float = 0.7
    openai_max_tokens: int = 150
    
    # Qdrant Configuration
    qdrant_url: str
    qdrant_api_key: Optional[str] = None
    qdrant_collection_name: str = "docgpt_collection"
    qdrant_prefer_grpc: bool = False
    
    # Document Processing Configuration
    chunk_size: int = 300
    chunk_overlap: int = 40
    max_file_size_mb: int = 10
    allowed_file_types: List[str] = ["application/pdf"]
    
    # Security Configuration
    jwt_secret_key: str = "your-secret-key-change-this"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 30
    
    # CORS Configuration
    frontend_url: Optional[str] = None
    allowed_origins: List[str] = ["http://localhost:3000"]
    
    # Rate Limiting
    rate_limit_requests: str = "10/minute"
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Create global settings instance
settings = Settings() 