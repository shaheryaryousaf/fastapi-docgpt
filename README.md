# 📄 FastAPI DocGPT - Enhanced Document Q&A System

A production-ready FastAPI backend for intelligent document question-answering using OpenAI embeddings and Qdrant vector database. Upload PDFs, ask questions, and get AI-powered answers from your documents.

## 🚀 **New Features & Improvements**

### ✨ **Enhanced Features**
- 🔒 **Security**: JWT authentication, rate limiting, input validation
- 📊 **Monitoring**: Comprehensive logging, metrics endpoint, health checks
- 🏗️ **Architecture**: Modular design with proper separation of concerns
- 🧪 **Testing**: Complete test suite with 11 test cases
- 🐳 **Deployment**: Docker support with docker-compose
- ⚙️ **Configuration**: Environment-based configuration management
- 🛡️ **Validation**: Enhanced input validation with Pydantic models
- 📈 **Performance**: Optimized error handling and processing times

### 🛠️ **Technical Improvements**
- **Structured Configuration**: Centralized settings management
- **Enhanced Error Handling**: Comprehensive error logging and user-friendly messages
- **Better Models**: Type-safe Pydantic models with validation
- **Security Layer**: JWT authentication and rate limiting (configurable)
- **Health Monitoring**: Detailed health checks with dependency status
- **Testing Suite**: Comprehensive test coverage
- **Docker Support**: Containerized deployment with multi-service setup

## 📦 **Project Structure**

```
fastapi-docgpt/
├── app.py                 # Main FastAPI application
├── utils.py              # Enhanced utility functions  
├── models.py             # Pydantic models for validation
├── config.py             # Configuration management
├── security.py           # Security and authentication
├── test_app.py           # Comprehensive test suite
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Multi-service setup
├── vercel.json          # Vercel deployment config
└── README.md            # This file
```

## 🚀 **Quick Start**

### **Option 1: Local Development**

1. **Clone and Setup**
   ```bash
   git clone <your-repo-url>
   cd fastapi-docgpt
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Environment Configuration**
   Create a `.env` file:
   ```env
   OPENAI_API_KEY=your-openai-api-key
   QDRANT_URL=http://localhost:6333
   QDRANT_API_KEY=your-qdrant-api-key  # Optional
   JWT_SECRET_KEY=your-secret-key
   DEBUG=true
   ```

3. **Run the Application**
   ```bash
   uvicorn app:app --reload
   ```

### **Option 2: Docker Deployment**

1. **Using Docker Compose (Recommended)**
   ```bash
   # Copy environment variables
   cp .env.example .env
   # Edit .env with your API keys
   
   # Start all services
   docker-compose up -d
   ```

2. **Manual Docker Build**
   ```bash
   docker build -t fastapi-docgpt .
   docker run -p 8000:8000 --env-file .env fastapi-docgpt
   ```

## 📋 **API Endpoints**

### **Core Endpoints**

| Method | Endpoint | Description | Authentication |
|--------|----------|-------------|----------------|
| `GET` | `/` | Health check with dependency status | ❌ |
| `GET` | `/health` | Detailed health check | ❌ |
| `GET` | `/metrics` | Application metrics | ❌ |
| `POST` | `/upload-pdf/` | Upload and process PDF | ⚠️ Optional |
| `POST` | `/ask-question/` | Ask questions about documents | ⚠️ Optional |

### **Enhanced Response Models**

#### Upload Response
```json
{
  "message": "PDF successfully processed and stored in vector DB",
  "document_id": "uuid-string",
  "chunks_processed": 25,
  "processing_time": 2.34,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### Question Response
```json
{
  "answer": "Your answer based on document content...",
  "processing_time": 1.2,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### Health Check Response
```json
{
  "status": "Success",
  "version": "1.0.0",
  "timestamp": "2024-01-01T12:00:00Z",
  "dependencies": {
    "qdrant": "connected",
    "openai": "configured",
    "environment": "development"
  }
}
```

## 🔧 **Configuration Options**

### **Environment Variables**

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | OpenAI API key | - | ✅ |
| `QDRANT_URL` | Qdrant database URL | - | ✅ |
| `QDRANT_API_KEY` | Qdrant API key | None | ❌ |
| `JWT_SECRET_KEY` | JWT signing key | "change-this" | ⚠️ |
| `DEBUG` | Debug mode | False | ❌ |
| `MAX_FILE_SIZE_MB` | Max upload size | 10 | ❌ |
| `CHUNK_SIZE` | Document chunk size | 300 | ❌ |
| `CHUNK_OVERLAP` | Chunk overlap | 40 | ❌ |
| `RATE_LIMIT_REQUESTS` | Rate limit | "10/minute" | ❌ |

### **Advanced Configuration**

The application uses a centralized configuration system. See `config.py` for all available settings:
- OpenAI model selection
- Embedding model configuration  
- Vector database settings
- Security parameters
- Logging configuration

## 🛡️ **Security Features**

### **Authentication (Optional)**
- JWT-based authentication
- Configurable token expiration
- Secure token validation

### **Rate Limiting**
- Configurable rate limits per endpoint
- IP-based limiting
- Prevents API abuse

### **Input Validation**
- File type validation (PDF only)
- File size limits
- Question length validation
- Sanitized error messages

## 🧪 **Testing**

Run the comprehensive test suite:

```bash
# Run all tests
pytest test_app.py -v

# Run with coverage
pytest test_app.py --cov=app --cov-report=html

# Run specific test categories
pytest test_app.py::TestHealthEndpoints -v
pytest test_app.py::TestFileUpload -v
pytest test_app.py::TestQuestionAnswering -v
```

**Test Coverage:**
- ✅ Health check endpoints
- ✅ File upload validation
- ✅ PDF processing workflow
- ✅ Question-answer functionality
- ✅ Error handling scenarios
- ✅ Input validation

## 📊 **Monitoring & Observability**

### **Logging**
- Structured logging with configurable levels
- Request/response tracking
- Performance metrics
- Error tracking with stack traces

### **Health Checks**
- Application health status
- Dependency health (Qdrant, OpenAI)
- Environment information
- Version tracking

### **Metrics**
- Processing times
- Request counts
- Configuration details
- System information

## 🐳 **Docker Deployment**

### **Production Deployment**
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  app:
    image: fastapi-docgpt:latest
    environment:
      - DEBUG=false
      - LOG_LEVEL=INFO
    restart: always
```

### **Environment Setup**
```bash
# Production environment
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## 🔄 **Migration from Original**

### **Breaking Changes**
- Enhanced response models (includes timestamps, processing times)
- Optional authentication (can be enabled by uncommenting decorators)
- Environment variable names standardized

### **Backward Compatibility**
- All original endpoints work the same way
- Same request formats
- Enhanced but compatible responses

## 🚀 **Performance Optimizations**

1. **Efficient Document Processing**
   - Configurable chunk sizes
   - Optimized PDF parsing
   - Memory-efficient file handling

2. **Vector Database Optimization**
   - Connection pooling
   - Batch operations
   - Persistent collections

3. **Caching Strategy**
   - Redis integration ready
   - Response caching capability
   - Embedding cache support

## 📈 **Scaling Considerations**

### **Horizontal Scaling**
- Stateless application design
- External vector database
- Load balancer ready

### **Vertical Scaling**
- Configurable resource limits
- Memory optimization
- CPU-efficient processing

## 🔍 **Troubleshooting**

### **Common Issues**

1. **Qdrant Connection Failed**
   ```bash
   # Check Qdrant status
   curl http://localhost:6333/health
   
   # Restart Qdrant
   docker-compose restart qdrant
   ```

2. **OpenAI API Issues**
   ```bash
   # Verify API key
   curl -H "Authorization: Bearer $OPENAI_API_KEY" \
        https://api.openai.com/v1/models
   ```

3. **File Upload Errors**
   - Check file size limits
   - Verify PDF format
   - Check disk space

### **Debug Mode**
```bash
# Enable debug logging
DEBUG=true LOG_LEVEL=DEBUG uvicorn app:app --reload
```

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## 📄 **License**

This project is licensed under the MIT License.

## 🙏 **Acknowledgments**

- FastAPI for the excellent web framework
- OpenAI for AI capabilities
- Qdrant for vector database
- LangChain for document processing

---

**Ready to build amazing document Q&A applications!** 🚀
