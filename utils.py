from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import os
import logging
import time
import uuid
from typing import List, Optional, Tuple
from config import settings

# Load environment variables (if needed)
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format=settings.log_format
)
logger = logging.getLogger(__name__)

# API keys and URLs from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


# Function to process PDF and split it into chunks
def process_pdf(pdf_path: str) -> Tuple[List, int]:
    """Process the PDF, split it into chunks, and return the chunks with count."""
    try:
        start_time = time.time()
        logger.info(f"Starting PDF processing for: {pdf_path}")
        
        # Validate file exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Check file size
        file_size = os.path.getsize(pdf_path)
        max_size = settings.max_file_size_mb * 1024 * 1024
        if file_size > max_size:
            raise ValueError(f"File size {file_size} exceeds maximum allowed size {max_size}")
        
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        if not pages:
            raise ValueError("No content found in PDF file")
        
        document_text = "".join([page.page_content for page in pages])
        
        if not document_text.strip():
            raise ValueError("PDF contains no readable text")

        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        chunks = text_splitter.create_documents([document_text])
        
        processing_time = time.time() - start_time
        logger.info(f"PDF processing completed in {processing_time:.2f}s. Created {len(chunks)} chunks")
        
        return chunks, len(chunks)
        
    except Exception as ex:
        logger.error(f"Failed to process PDF {pdf_path}: {str(ex)}")
        raise


# Function to send document chunks (with embeddings) to the Qdrant vector database
def send_to_qdrant(documents: List, embedding_model: OpenAIEmbeddings, document_id: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """Send the document chunks to the Qdrant vector database."""
    try:
        start_time = time.time()
        doc_id = document_id or str(uuid.uuid4())
        
        logger.info(f"Storing {len(documents)} documents in Qdrant with ID: {doc_id}")
        
        # Add metadata to documents
        for i, doc in enumerate(documents):
            doc.metadata.update({
                "document_id": doc_id,
                "chunk_index": i,
                "timestamp": time.time()
            })
        
        qdrant = Qdrant.from_documents(
            documents,
            embedding_model,
            url=settings.qdrant_url,
            prefer_grpc=settings.qdrant_prefer_grpc,
            api_key=settings.qdrant_api_key,
            collection_name=settings.qdrant_collection_name,
            force_recreate=False  # Don't recreate collection every time
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Successfully stored documents in Qdrant in {processing_time:.2f}s")
        
        return True, doc_id
        
    except Exception as ex:
        logger.error(f"Failed to store data in the vector DB: {str(ex)}")
        return False, None


# Function to initialize the Qdrant client and return the vector store object
def qdrant_client() -> Optional[Qdrant]:
    """Initialize Qdrant client and return the vector store."""
    try:
        logger.info("Initializing Qdrant client")
        
        embedding_model = OpenAIEmbeddings(
            openai_api_key=settings.openai_api_key, 
            model=settings.openai_embedding_model
        )
        
        qdrant_client_instance = QdrantClient(
            url=settings.qdrant_url, 
            api_key=settings.qdrant_api_key
        )
        
        qdrant_store = Qdrant(
            client=qdrant_client_instance,
            collection_name=settings.qdrant_collection_name,
            embeddings=embedding_model
        )
        
        logger.info("Qdrant client initialized successfully")
        return qdrant_store
        
    except Exception as ex:
        logger.error(f"Failed to initialize Qdrant client: {str(ex)}")
        return None


# Function to handle question answering using the Qdrant vector store and GPT
def qa_ret(qdrant_store: Qdrant, input_query: str) -> Tuple[str, Optional[float]]:
    """Retrieve relevant documents and generate a response from the AI model."""
    try:
        start_time = time.time()
        logger.info(f"Processing question: {input_query[:100]}...")
        
        template = """
        Instructions:
            You are trained to extract answers from the given Context and the User's Question. Your response must be based on semantic understanding, which means even if the wording is not an exact match, infer the closest possible meaning from the Context. 

            Key Points to Follow:
            - **Precise Answer Length**: The answer must be between a minimum of 40 words and a maximum of 100 words.
            - **Strict Answering Rules**: Do not include any unnecessary text. The answer should be concise and focused directly on the question.
            - **Professional Language**: Do not use any abusive or prohibited language. Always respond in a polite and gentle tone.
            - **No Personal Information Requests**: Do not ask for personal information from the user at any point.
            - **Concise & Understandable**: Provide the most concise, clear, and understandable answer possible.
            - **Semantic Similarity**: If exact wording isn't available in the Context, use your semantic understanding to infer the answer. If there are semantically related phrases, use them to generate a precise response. Use natural language understanding to interpret closely related words or concepts.
            - **Unavailable Information**: If the answer is genuinely not found in the Context, politely apologize and inform the user that the specific information is not available in the provided context.

            Context:
            {context}

            **User's Question:** {question}

            Respond in a polite, professional, and concise manner.
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        retriever = qdrant_store.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 4}
        )

        setup_and_retrieval = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        )

        model = ChatOpenAI(
            model_name=settings.openai_model,
            temperature=settings.openai_temperature,
            openai_api_key=settings.openai_api_key,
            max_tokens=settings.openai_max_tokens
        )

        output_parser = StrOutputParser()

        rag_chain = setup_and_retrieval | prompt | model | output_parser
        response = rag_chain.invoke(input_query)
        
        processing_time = time.time() - start_time
        logger.info(f"Question answered in {processing_time:.2f}s")
        
        return response, processing_time

    except Exception as ex:
        logger.error(f"Error in QA processing: {str(ex)}")
        return f"Error: {str(ex)}", None


def validate_file_type(file_content_type: str) -> bool:
    """Validate if the uploaded file type is allowed."""
    return file_content_type in settings.allowed_file_types


def get_file_size(file_path: str) -> int:
    """Get file size in bytes."""
    return os.path.getsize(file_path)


def cleanup_temp_file(file_path: str) -> None:
    """Safely remove temporary file."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
    except Exception as ex:
        logger.warning(f"Failed to cleanup temporary file {file_path}: {str(ex)}")
