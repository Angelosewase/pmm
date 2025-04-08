import pytest
from fastapi.testclient import TestClient
from app.api.endpoints import app
import json
import os
from datetime import datetime

client = TestClient(app)

# Test data
SAMPLE_PREDICTION_INPUT = {
    "engine_rpm": 2500.0,
    "lub_oil_pressure": 45.5,
    "fuel_pressure": 35.0,
    "coolant_pressure": 15.0,
    "lub_oil_temp": 85.0,
    "coolant_temp": 95.0,
    "timestamp": datetime.now().isoformat()
}

SAMPLE_BATCH_INPUT = {
    "predictions": [
        SAMPLE_PREDICTION_INPUT,
        {
            **SAMPLE_PREDICTION_INPUT,
            "engine_rpm": 3000.0,
            "lub_oil_temp": 90.0
        }
    ]
}

# Get API key from environment
API_KEY = os.getenv("API_KEY", "your-secret-api-key")
HEADERS = {"X-API-Key": API_KEY}

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"

def test_predict():
    """Test the prediction endpoint."""
    response = client.post(
        "/predict",
        json=SAMPLE_PREDICTION_INPUT,
        headers=HEADERS
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert "status" in data
    assert "prediction_id" in data
    assert "feature_importance" in data

def test_batch_predict():
    """Test the batch prediction endpoint."""
    response = client.post(
        "/predict/batch",
        json=SAMPLE_BATCH_INPUT,
        headers=HEADERS
    )
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "batch_id" in data
    assert "summary" in data
    assert len(data["predictions"]) == len(SAMPLE_BATCH_INPUT["predictions"])

def test_invalid_api_key():
    """Test authentication with invalid API key."""
    response = client.post(
        "/predict",
        json=SAMPLE_PREDICTION_INPUT,
        headers={"X-API-Key": "invalid-key"}
    )
    assert response.status_code == 403

def test_train_model():
    """Test the model training endpoint."""
    response = client.post(
        "/train",
        headers=HEADERS
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "accepted"

def test_get_prediction():
    """Test retrieving a prediction by ID."""
    # First make a prediction to get an ID
    predict_response = client.post(
        "/predict",
        json=SAMPLE_PREDICTION_INPUT,
        headers=HEADERS
    )
    prediction_id = predict_response.json()["prediction_id"]
    
    # Then retrieve it
    response = client.get(
        f"/predictions/{prediction_id}",
        headers=HEADERS
    )
    assert response.status_code == 200
    data = response.json()
    assert data["prediction_id"] == prediction_id

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 