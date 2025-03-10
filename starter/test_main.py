import pytest
from fastapi.testclient import TestClient
from starter.main import app

client = TestClient(app)

def test_get_root():
    """
    Test the GET endpoint on the root domain.
    """

    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the inference API!"}

