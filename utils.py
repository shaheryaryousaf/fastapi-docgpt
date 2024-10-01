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

# Load environment variables (if needed)
from dotenv import load_dotenv
load_dotenv()

# API keys and URLs from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


# Function to process PDF and split it into chunks
def process_pdf(pdf_path):
    """Process the PDF, split it into chunks, and return the chunks."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    document_text = "".join([page.page_content for page in pages])

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # Adjust as needed
        chunk_overlap=40  # Adjust as needed
    )
    chunks = text_splitter.create_documents([document_text])

    return chunks


# Function to send document chunks (with embeddings) to the Qdrant vector database
def send_to_qdrant(documents, embedding_model):
    """Send the document chunks to the Qdrant vector database."""
    try:
        qdrant = Qdrant.from_documents(
            documents,
            embedding_model,
            url=QDRANT_URL,
            prefer_grpc=False,
            api_key=QDRANT_API_KEY,
            collection_name="xeven_chatbot",  # Replace with your collection name
            force_recreate=True  # Create a fresh collection every time
        )
        return True
    except Exception as ex:
        print(f"Failed to store data in the vector DB: {str(ex)}")
        return False


# Function to initialize the Qdrant client and return the vector store object
def qdrant_client():
    """Initialize Qdrant client and return the vector store."""
    embedding_model = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002"
    )
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    qdrant_store = Qdrant(
        client=qdrant_client,
        collection_name="xeven_chatbot",
        embeddings=embedding_model
    )
    return qdrant_store


# Function to handle question answering using the Qdrant vector store and GPT
def qa_ret(qdrant_store, input_query):
    """Retrieve relevant documents and generate a response from the AI model."""
    try:
        template = """
        Instructions:
            You are trained to extract answers from the given Context and the User's Question. Your response must be based on semantic understanding, which means even if the wording is not an exact match, infer the closest possible meaning from the Context. 

            Key Points to Follow:
            - **Precise Answer Length**: The answer must be between a minimum of 40 words and a maximum of 100 words.
            - **Strict Answering Rules**: Do not include any unnecessary text. The answer should be concise and focused directly on the question.
            - **Professional Language**: Do not use any abusive or prohibited language. Always respond in a polite and gentle tone.
            - **No Personal Information Requests**: Do not ask for personal information from the user at any point.
            - **Concise & Understandable**: Provide the most concise, clear, and understandable answer possible.
            - **Semantic Similarity**: If exact wording isn’t available in the Context, use your semantic understanding to infer the answer. If there are semantically related phrases, use them to generate a precise response. Use natural language understanding to interpret closely related words or concepts.
            - **Unavailable Information**: If the answer is genuinely not found in the Context, politely apologize and inform the user that the specific information is not available in the provided context.

            Context:
            {context}

            **User's Question:** {question}

            Respond in a polite, professional, and concise manner.
        """
        prompt = ChatPromptTemplate.from_template(template)
        retriever = qdrant_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        setup_and_retrieval = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        )

        model = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.7,
            openai_api_key=OPENAI_API_KEY,
            max_tokens=150
        )

        output_parser = StrOutputParser()

        rag_chain = setup_and_retrieval | prompt | model | output_parser
        response = rag_chain.invoke(input_query)
        return response

    except Exception as ex:
        return f"Error: {str(ex)}"
