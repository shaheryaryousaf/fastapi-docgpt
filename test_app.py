import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import tempfile
import os
from app import app

client = TestClient(app)

class TestHealthEndpoints:
    def test_health_check(self):
        """Test the health check endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "Success" or data["status"] == "Error"
        assert "version" in data
        assert "timestamp" in data

    def test_health_endpoint(self):
        """Test the /health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "Success" or data["status"] == "Error"

    def test_metrics_endpoint(self):
        """Test the metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "app_name" in data
        assert "version" in data
        assert "settings" in data

class TestFileUpload:
    def test_upload_invalid_file_type(self):
        """Test uploading an invalid file type."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            temp_file.write(b"test content")
            temp_file_path = temp_file.name

        try:
            with open(temp_file_path, "rb") as f:
                response = client.post(
                    "/upload-pdf/",
                    files={"file": ("test.txt", f, "text/plain")}
                )
            assert response.status_code == 400
            assert "Invalid file type" in response.json()["detail"]
        finally:
            os.unlink(temp_file_path)

    @patch('app.process_pdf')
    @patch('app.send_to_qdrant')
    def test_upload_pdf_success(self, mock_send_to_qdrant, mock_process_pdf):
        """Test successful PDF upload."""
        # Mock the functions
        mock_process_pdf.return_value = ([], 5)  # chunks, count
        mock_send_to_qdrant.return_value = (True, "test-doc-id")

        # Create a fake PDF file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(b"%PDF-1.4 fake pdf content")
            temp_file_path = temp_file.name

        try:
            with open(temp_file_path, "rb") as f:
                response = client.post(
                    "/upload-pdf/",
                    files={"file": ("test.pdf", f, "application/pdf")}
                )
            
            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "PDF successfully processed and stored in vector DB"
            assert "document_id" in data
            assert "chunks_processed" in data
        finally:
            os.unlink(temp_file_path)

class TestQuestionAnswering:
    def test_ask_question_invalid_input(self):
        """Test asking a question with invalid input."""
        response = client.post(
            "/ask-question/",
            json={"question": ""}  # Empty question
        )
        assert response.status_code == 422  # Validation error

    def test_ask_question_short_input(self):
        """Test asking a question with too short input."""
        response = client.post(
            "/ask-question/",
            json={"question": "hi"}  # Too short
        )
        assert response.status_code == 422  # Validation error

    @patch('app.qdrant_client')
    @patch('app.qa_ret')
    def test_ask_question_success(self, mock_qa_ret, mock_qdrant_client):
        """Test successful question answering."""
        # Mock the functions
        mock_qdrant_client.return_value = MagicMock()
        mock_qa_ret.return_value = ("This is a test answer", 1.5)

        response = client.post(
            "/ask-question/",
            json={"question": "What is this document about?"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "This is a test answer"
        assert "processing_time" in data
        assert "timestamp" in data

    @patch('app.qdrant_client')
    def test_ask_question_qdrant_failure(self, mock_qdrant_client):
        """Test question answering when Qdrant connection fails."""
        mock_qdrant_client.return_value = None

        response = client.post(
            "/ask-question/",
            json={"question": "What is this document about?"}
        )
        
        assert response.status_code == 500
        assert "Failed to connect to vector database" in response.json()["detail"]

class TestValidation:
    def test_question_whitespace_validation(self):
        """Test that questions with only whitespace are rejected."""
        response = client.post(
            "/ask-question/",
            json={"question": "   "}  # Only whitespace
        )
        assert response.status_code == 422

    def test_question_too_long(self):
        """Test that very long questions are rejected."""
        long_question = "What is this? " * 100  # Very long question
        response = client.post(
            "/ask-question/",
            json={"question": long_question}
        )
        assert response.status_code == 422

if __name__ == "__main__":
    pytest.main([__file__]) 