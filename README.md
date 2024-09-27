# 📄 Document GPT - FastAPI Backend

This FastAPI backend serves as the core API for handling document uploads, processing PDF files, embedding document content into a vector database (Qdrant), and allowing users to ask questions based on the uploaded document. The AI model uses OpenAI's embeddings to generate intelligent responses from the document content.

### 🛠️ Features
 - **PDF Upload:** Upload PDF files to be processed and stored in a vector database (Qdrant) for querying.
 - **Question & Answer System:** Users can ask questions based on the content of the uploaded PDF.
 - **API Documentation:** Automatic API documentation available through Swagger at /docs.

### 📦 Libraries Used
 - **FastAPI:** For building the web API.
 - **Qdrant Client:** For storing and retrieving document embeddings.
 - **LangChain:** For handling PDF processing and embeddings.
 - **OpenAI:** For generating embeddings and AI model responses.
 - **PyPDFLoader:** For extracting text from PDF files.
 - **CORS Middleware:** For handling Cross-Origin Resource Sharing (CORS) to allow frontend requests from different domains.
 - **dotenv:** For managing environment variables (e.g., API keys).

### 🗂️ Project Structure
 - ```app.py```: Main FastAPI application file containing the API endpoints for PDF upload and question-answer system.
 - ```utils.py```: Contains utility functions for processing PDF files, sending embeddings to the vector DB, and retrieving answers from the embeddings.
 - **Environment Variables:** API keys for OpenAI and Qdrant are managed through environment variables using ```.env``` file.

## 🚀 Getting Started

### Prerequisites
Before setting up the FastAPI backend, ensure you have the following installed:
 - Python 3.7+
 - Pip (Python package manager)
 - Qdrant (a vector database, can be run locally or via a managed service)
 - OpenAI API Key (for generating embeddings and responses)
 - Virtual environment (optional but recommended)

### 🛠️ Installation & Setup
Follow these steps to set up the FastAPI backend on your local machine:

#### Step 1: Clone the Repository
```
git clone <your-repo-url>
cd <your-repo-name>
```

#### Step 2: Set Up a Virtual Environment
It is recommended to create a virtual environment to manage the dependencies:
```
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Step 3: Install Dependencies
Install the required dependencies using ```pip```:
```
pip install -r requirements.txt
```
If there is no requirements.txt file, manually install these packages:
```
pip install fastapi qdrant-client langchain pydantic uvicorn python-dotenv openai
```

#### Step 4: Set Up Environment Variables
Create a ```.env``` file in the root directory and add the necessary API keys for OpenAI and Qdrant:
```
OPENAI_API_KEY=your-openai-api-key
QDRANT_URL=your-qdrant-url
QDRANT_API_KEY=your-qdrant-api-key
```
 - ```OPENAI_API_KEY:``` The API key for accessing OpenAI services.
 - ```QDRANT_URL:``` The URL to your Qdrant instance.
 - ```QDRANT_API_KEY:``` The API key for Qdrant (if required).

#### Step 5: Run the FastAPI Application
Start the FastAPI server locally by running the following command:
```
uvicorn app:app --reload
```
This will start the server at ```http://127.0.0.1:8000/```.

#### Step 6: Test the API on Swagger UI
FastAPI automatically generates API documentation, accessible through Swagger.
Open your browser and navigate to: ```http://127.0.0.1:8000/docs```

Here you can test both API endpoints directly:
 - ```/upload-pdf/```: Upload a PDF file for processing and storage in Qdrant.
 - ```/ask-question/```: Ask a question based on the uploaded PDF's content.

## 📄 API Endpoints
#### 1. Upload PDF - ```/upload-pdf/``` [POST]
Uploads a PDF file, processes it, creates embeddings, and stores them in Qdrant.

**Request:**
 - **Method:** POST
 - **Content Type:** ```multipart/form-data```
 - **Body:** PDF file to upload.

**Response:**
**Success:** ```{ "message": "PDF successfully processed and stored in vector DB" }```
**Error:** ```{ "detail": "Failed to process PDF: <error-message>" }```

#### 2. Ask Question - ```/ask-question/``` [POST]
Accepts a question and returns an answer based on the content stored in the vector database from the uploaded PDF.

**Request:**
 - **Method:** POST
 - **Content Type:** application/json
 - **Body:**
    ```
    {
    "question": "What is the summary of this document?"
    }
    ```
**Response:**

**Success:** ```{ "answer": "<response-from-the-document>" }```
**Error:** ```{ "detail": "Failed to retrieve answer: <error-message>" }```

#### 3. Health Check - ```/``` [GET]
A simple health check endpoint to verify that the API is up and running.

**Response:**
 - **Success:** ```{ "status": "Success" }```

## 🧑‍💻 Utils Overview
The ```utils.py``` file contains utility functions that handle core logic for processing PDFs, sending embeddings to Qdrant, and retrieving answers from stored documents.

### Key Functions in ```utils.py```:

1. ```process_pdf(pdf_path)```:

    - Extracts the text from the PDF and splits it into smaller chunks.
    - **Input:** Path to the PDF file.
    - **Returns:** A list of text chunks from the PDF.

2. ```send_to_qdrant(documents, embedding_model)```:

    - Sends the processed document chunks to Qdrant for storage after creating embeddings.
    - **Input:** List of document chunks and an embedding model.
    - **Returns:** True if successful, False if there’s an error.

3. ```qdrant_client()```:

    - Initializes and returns a Qdrant client for interacting with the vector database.
    - **Returns:** A configured Qdrant vector store.

4. ```qa_ret(qdrant_store, input_query)```:

    - Handles question-answering by retrieving the relevant content from Qdrant and generating a response using OpenAI's GPT model.
    - **Input:** The Qdrant vector store and the user’s question.
    - **Returns:** A generated response based on the document’s context.

## 🧪 Testing the Application
#### Test PDF Upload
 - Start the FastAPI server (```uvicorn app:app --reload```).
 - Use Swagger at ```http://127.0.0.1:8000/docs``` to upload a PDF.
 - After the PDF is processed, use the ```/ask-question/``` endpoint to ask a question based on the uploaded content.

## ⚙️ Deployment Considerations
 - Ensure environment variables are properly set in your production environment for API keys.
 - Use a scalable deployment method like **Docker** or deploy to a cloud service like **AWS**, **Google Cloud**, or **Heroku**.
 - You can deploy Qdrant as a managed service or host your own instance, depending on your requirements.
