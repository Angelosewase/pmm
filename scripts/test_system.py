import requests
import pandas as pd
import json
from datetime import datetime
import time
import os
from typing import Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MaintenanceAPIClient:
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = None):
        self.base_url = base_url
        self.api_key = api_key or os.getenv("API_KEY", "your-secret-api-key")
        self.headers = {"X-API-Key": self.api_key}
    
    def health_check(self) -> Dict[str, Any]:
        """Check if the API is healthy."""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a single prediction."""
        response = requests.post(
            f"{self.base_url}/predict",
            json=data,
            headers=self.headers
        )
        return response.json()
    
    def batch_predict(self, data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make batch predictions."""
        response = requests.post(
            f"{self.base_url}/predict/batch",
            json={"predictions": data_list},
            headers=self.headers
        )
        return response.json()
    
    def train_model(self) -> Dict[str, Any]:
        """Trigger model training."""
        response = requests.post(
            f"{self.base_url}/train",
            headers=self.headers
        )
        return response.json()
    
    def get_prediction(self, prediction_id: str) -> Dict[str, Any]:
        """Retrieve a specific prediction."""
        response = requests.get(
            f"{self.base_url}/predictions/{prediction_id}",
            headers=self.headers
        )
        return response.json()

def load_test_data(file_path: str = "data/vehicle_maintenance_data.csv") -> pd.DataFrame:
    """Load test data from CSV file."""
    return pd.read_csv(file_path)

def prepare_prediction_input(row: pd.Series) -> Dict[str, Any]:
    """Convert a row of data to prediction input format."""
    return {
        "engine_rpm": float(row["Engine rpm"]),
        "lub_oil_pressure": float(row["Lub oil pressure"]),
        "fuel_pressure": float(row["Fuel pressure"]),
        "coolant_pressure": float(row["Coolant pressure"]),
        "lub_oil_temp": float(row["lub oil temp"]),
        "coolant_temp": float(row["Coolant temp"]),
        "timestamp": datetime.now().isoformat()
    }

def main():
    # Initialize client
    client = MaintenanceAPIClient()
    
    # Check health
    logger.info("Checking API health...")
    health_status = client.health_check()
    logger.info(f"Health status: {health_status}")
    
    # Load test data
    logger.info("Loading test data...")
    df = load_test_data()
    logger.info(f"Loaded {len(df)} records")
    
    # Test single prediction
    logger.info("Testing single prediction...")
    single_input = prepare_prediction_input(df.iloc[0])
    prediction = client.predict(single_input)
    logger.info(f"Single prediction result: {prediction}")
    
    # Test batch prediction
    logger.info("Testing batch prediction...")
    batch_inputs = [prepare_prediction_input(row) for _, row in df.head(5).iterrows()]
    batch_results = client.batch_predict(batch_inputs)
    logger.info(f"Batch prediction summary: {batch_results['summary']}")
    
    # Test model training
    logger.info("Testing model training...")
    training_response = client.train_model()
    logger.info(f"Training response: {training_response}")
    
    # Test prediction retrieval
    logger.info("Testing prediction retrieval...")
    prediction_id = prediction["prediction_id"]
    retrieved_prediction = client.get_prediction(prediction_id)
    logger.info(f"Retrieved prediction: {retrieved_prediction}")

if __name__ == "__main__":
    main() 