from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os

# Import the necessary functions from utils.py
from utils import process_pdf, send_to_qdrant, qdrant_client, qa_ret, OpenAIEmbeddings

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from your React app (adjust domain if necessary)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (POST, GET, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Define a model for the question API
class QuestionRequest(BaseModel):
    question: str

# Endpoint to upload a PDF and process it, sending to Qdrant
@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Endpoint to upload a PDF file, process it, and store in the vector DB.
    """
    try:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.file.read())
            temp_file_path = temp_file.name

        # Process the PDF to get document chunks and embeddings
        document_chunks = process_pdf(temp_file_path)

        # Create the embedding model (e.g., OpenAIEmbeddings)
        embedding_model = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"),  # Assuming you're using env vars
            model="text-embedding-ada-002"
        )

        # Send the document chunks (with embeddings) to Qdrant
        success = send_to_qdrant(document_chunks, embedding_model)

        # Remove the temporary file after processing
        os.remove(temp_file_path)

        if success:
            return {"message": "PDF successfully processed and stored in vector DB"}
        else:
            raise HTTPException(status_code=500, detail="Failed to store PDF in vector DB")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

# Endpoint to ask a question and retrieve the answer from the vector DB
@app.post("/ask-question/")
async def ask_question(question_request: QuestionRequest):
    """
    Endpoint to ask a question and retrieve a response from the stored document content.
    """
    try:
        # Retrieve the Qdrant vector store (assuming qdrant_client() gives you access to it)
        qdrant_store = qdrant_client()

        # Get the question from the request body
        question = question_request.question

        # Use the question-answer retrieval function to get the response
        response = qa_ret(qdrant_store, question)

        return {"answer": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve answer: {str(e)}")

# A simple health check endpoint
@app.get("/")
async def health_check():
    return {"status": "Success"}

