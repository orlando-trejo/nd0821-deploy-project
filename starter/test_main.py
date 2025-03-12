import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_get_root():
    """
    Test the GET endpoint on the root domain.
    """

    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the inference API!"}

def test_post_predict_low():
    """
    Test the POST /predict endpoint with data likely leading to a '0' prediction.
    """
    sample_record = {
        "age": 25,
        "workclass": "Private",
        "fnlgt": 226802,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Never-married",
        "occupation": "Machine-op-inspct",
        "relationship": "Own-child",
        "race": "Black",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    response = client.post("/predict", json=sample_record)
    assert response.status_code == 200
    assert "prediction" in response.json()
    # Typically '0' indicates <=50K. Adjust to your model’s behavior.
    assert response.json()["prediction"] in [0, 1]


def test_post_predict_high():
    """
    Test the POST /predict endpoint with data likely leading to a '1' prediction.
    """
    sample_record = {
        "age": 45,
        "workclass": "Private",
        "fnlgt": 300000,
        "education": "Doctorate",
        "education-num": 16,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 14000,
        "capital-loss": 0,
        "hours-per-week": 60,
        "native-country": "United-States"
    }
    response = client.post("/predict", json=sample_record)
    assert response.status_code == 200
    assert "prediction" in response.json()
    # Typically '1' indicates >50K. Adjust to your model’s behavior.
    assert response.json()["prediction"] in [0, 1]