from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile
import os
import time
import logging
from typing import Optional

# Import the necessary functions and models
from utils import (
    process_pdf, send_to_qdrant, qdrant_client, qa_ret, 
    OpenAIEmbeddings, validate_file_type, cleanup_temp_file
)
from models import (
    QuestionRequest, QuestionResponse, UploadResponse, 
    ErrorResponse, HealthResponse
)
from config import settings
from security import limiter, verify_token

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format=settings.log_format
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="A FastAPI backend for document-based question answering using AI",
    debug=settings.debug
)

# Add rate limiter
app.state.limiter = limiter

# Configure CORS
allowed_origins = settings.allowed_origins.copy()
if settings.frontend_url:
    allowed_origins.append(settings.frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            error_code="INTERNAL_ERROR"
        ).dict()
    )

# Endpoint to upload a PDF and process it, sending to Qdrant
@app.post("/upload-pdf/", response_model=UploadResponse)
@limiter.limit(settings.rate_limit_requests)
async def upload_pdf(
    request: Request,
    file: UploadFile = File(...),
    # user: str = Depends(verify_token)  # Uncomment to enable authentication
):
    """
    Endpoint to upload a PDF file, process it, and store in the vector DB.
    """
    start_time = time.time()
    temp_file_path = None
    
    try:
        logger.info(f"PDF upload started: {file.filename}")
        
        # Validate file type
        if not validate_file_type(file.content_type):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type. Allowed types: {settings.allowed_file_types}"
            )
        
        # Validate file size
        if file.size and file.size > settings.max_file_size_mb * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds maximum allowed size of {settings.max_file_size_mb}MB"
            )

        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Process the PDF to get document chunks and embeddings
        document_chunks, chunks_count = process_pdf(temp_file_path)

        # Create the embedding model
        embedding_model = OpenAIEmbeddings(
            openai_api_key=settings.openai_api_key,
            model=settings.openai_embedding_model
        )

        # Send the document chunks (with embeddings) to Qdrant
        success, document_id = send_to_qdrant(document_chunks, embedding_model)

        if not success:
            raise HTTPException(
                status_code=500, 
                detail="Failed to store PDF in vector DB"
            )

        processing_time = time.time() - start_time
        logger.info(f"PDF upload completed successfully in {processing_time:.2f}s")

        return UploadResponse(
            message="PDF successfully processed and stored in vector DB",
            document_id=document_id,
            chunks_processed=chunks_count,
            processing_time=processing_time
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process PDF: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to process PDF: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if temp_file_path:
            cleanup_temp_file(temp_file_path)

# Endpoint to ask a question and retrieve the answer from the vector DB
@app.post("/ask-question/", response_model=QuestionResponse)
@limiter.limit(settings.rate_limit_requests)
async def ask_question(
    request: Request,
    question_request: QuestionRequest,
    # user: str = Depends(verify_token)  # Uncomment to enable authentication
):
    """
    Endpoint to ask a question and retrieve a response from the stored document content.
    """
    try:
        logger.info(f"Question received: {question_request.question[:100]}...")
        
        # Retrieve the Qdrant vector store
        qdrant_store = qdrant_client()
        if not qdrant_store:
            raise HTTPException(
                status_code=500,
                detail="Failed to connect to vector database"
            )

        # Use the question-answer retrieval function to get the response
        response, processing_time = qa_ret(qdrant_store, question_request.question)

        logger.info("Question answered successfully")
        
        return QuestionResponse(
            answer=response,
            processing_time=processing_time
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve answer: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to retrieve answer: {str(e)}"
        )

# Enhanced health check endpoint
@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check endpoint with dependency status."""
    try:
        # Check Qdrant connection
        qdrant_status = "connected"
        try:
            qdrant_store = qdrant_client()
            if not qdrant_store:
                qdrant_status = "disconnected"
        except Exception:
            qdrant_status = "error"
        
        # Check OpenAI API key
        openai_status = "configured" if settings.openai_api_key else "not_configured"
        
        dependencies = {
            "qdrant": qdrant_status,
            "openai": openai_status,
            "environment": "production" if not settings.debug else "development"
        }
        
        return HealthResponse(
            status="Success",
            version=settings.app_version,
            dependencies=dependencies
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="Error",
            version=settings.app_version
        )

# New endpoint to get application metrics
@app.get("/metrics")
async def get_metrics():
    """Get application metrics."""
    return {
        "app_name": settings.app_name,
        "version": settings.app_version,
        "uptime": "N/A",  # Could implement actual uptime tracking
        "total_requests": "N/A",  # Could implement request counting
        "settings": {
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "max_file_size_mb": settings.max_file_size_mb,
            "openai_model": settings.openai_model
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )

