from fastapi import FastAPI, HTTPException, Depends, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import logging
import json
from datetime import datetime
import hashlib
import os
import time
import uuid
from app.core.local_config import local_settings
from app.preprocessing.data_processor import DataProcessor
from app.models.ml_model import MaintenanceModel
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=local_settings.PROJECT_NAME,
    description="Vehicle Predictive Maintenance API (Local Version)",
    version=local_settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=local_settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
data_processor = DataProcessor(local_settings.DATA_PATH)
model = MaintenanceModel(local_settings.MODEL_PATH)

# Simple file-based cache implementation
class FileCache:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get(self, key):
        file_path = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(file_path):
            # Check if cache is expired
            if time.time() - os.path.getmtime(file_path) > local_settings.PREDICTION_CACHE_TTL:
                return None
            with open(file_path, 'r') as f:
                return json.load(f)
        return None
    
    def set(self, key, value, expire=None):
        file_path = os.path.join(self.cache_dir, f"{key}.json")
        with open(file_path, 'w') as f:
            json.dump(value, f)
        return True

# Initialize file cache
cache = FileCache(local_settings.CACHE_DIR)

# Pydantic models for request/response
class PredictionInput(BaseModel):
    engine_rpm: float = Field(..., description="Engine RPM reading")
    lub_oil_pressure: float = Field(..., description="Lubrication oil pressure")
    fuel_pressure: float = Field(..., description="Fuel pressure")
    coolant_pressure: float = Field(..., description="Coolant pressure")
    lub_oil_temp: float = Field(..., description="Lubrication oil temperature")
    coolant_temp: float = Field(..., description="Coolant temperature")
    timestamp: Optional[datetime] = Field(None, description="Timestamp of the reading")

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    status: str
    feature_importance: Optional[Dict[str, float]]
    prediction_id: str
    timestamp: datetime

class ErrorResponse(BaseModel):
    detail: str
    timestamp: datetime
    error_code: str

class BatchPredictionInput(BaseModel):
    predictions: List[PredictionInput]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    batch_id: str
    summary: Dict[str, Any]

# API key dependency
async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != local_settings.API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    return x_api_key

def generate_cache_key(input_data: Dict[str, Any]) -> str:
    """Generate a cache key from input data."""
    sorted_items = sorted(input_data.items())
    return hashlib.md5(json.dumps(sorted_items).encode()).hexdigest()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check if model is loaded
        if model.model is None:
            model.load_model()
        
        # Check if data can be loaded
        test_data = data_processor.load_data()
        if test_data is None or len(test_data) == 0:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": "Data loading failed"}
            )
        
        return {
            "status": "healthy",
            "version": local_settings.VERSION,
            "timestamp": datetime.now().isoformat(),
            "model_loaded": model.model is not None,
            "data_available": True
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.post("/train", dependencies=[Depends(verify_api_key)])
async def train_model(background_tasks: BackgroundTasks):
    """Train the model with current data."""
    try:
        # Define a background task for training
        def train_model_task():
            try:
                # Load and preprocess data
                df = data_processor.load_data()
                processed_df = data_processor.preprocess_data(df)
                
                # Split data
                X_train, X_test, y_train, y_test = data_processor.split_data(processed_df)
                
                # Train model
                model.train(X_train, y_train)
                
                # Evaluate model
                metrics = model.evaluate(X_test, y_test)
                
                logger.info(f"Model training completed with metrics: {metrics}")
                return metrics
            except Exception as e:
                logger.error(f"Error in model training task: {str(e)}")
                raise
        
        # Add task to background tasks
        background_tasks.add_task(train_model_task)
        
        return {
            "status": "training_started",
            "message": "Model training has been started in the background",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error starting model training: {str(e)}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                detail=str(e),
                timestamp=datetime.now(),
                error_code="TRAINING_ERROR"
            ).dict()
        )

@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
async def predict(input_data: PredictionInput):
    """Make predictions for new data."""
    try:
        start_time = time.time()
        
        # Convert input to dict
        input_dict = input_data.dict()
        
        # Generate cache key
        cache_key = generate_cache_key(input_dict)
        
        # Check cache
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info(f"Cache hit for key: {cache_key}")
            cached_result['timestamp'] = datetime.now()
            return PredictionResponse(**cached_result)
        
        # Prepare input data
        X = data_processor.prepare_prediction_data(input_dict)
        
        # Ensure model is loaded
        if model.model is None:
            model.load_model()
        
        # Make prediction
        predictions, probabilities = model.predict(X)
        
        # Get prediction and probability
        prediction = int(predictions[0])
        probability = float(probabilities[0][1])
        
        # Get feature importance
        feature_importance = model.feature_importance()
        
        # Generate prediction ID
        prediction_id = str(uuid.uuid4())
        
        # Prepare response
        response = PredictionResponse(
            prediction=prediction,
            probability=probability,
            status="Maintenance Required" if prediction == 1 else "Normal",
            feature_importance=feature_importance,
            prediction_id=prediction_id,
            timestamp=datetime.now()
        )
        
        # Store in cache
        cache.set(cache_key, response.dict())
        
        logger.info(f"Prediction completed in {time.time() - start_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                detail=str(e),
                timestamp=datetime.now(),
                error_code="PREDICTION_ERROR"
            ).dict()
        )

@app.post("/predict/batch", response_model=BatchPredictionResponse, dependencies=[Depends(verify_api_key)])
async def batch_predict(input_data: BatchPredictionInput):
    """Make predictions for multiple inputs."""
    try:
        predictions = []
        maintenance_required = 0
        total_probability = 0
        
        for single_input in input_data.predictions:
            # Make prediction
            result = await predict(single_input)
            predictions.append(result)
            
            # Update statistics
            if result.prediction == 1:
                maintenance_required += 1
            total_probability += result.probability
        
        # Calculate summary statistics
        total_predictions = len(predictions)
        summary = {
            "total_predictions": total_predictions,
            "maintenance_required": maintenance_required,
            "maintenance_required_percentage": (maintenance_required / total_predictions) * 100 if total_predictions > 0 else 0,
            "average_probability": total_probability / total_predictions if total_predictions > 0 else 0
        }
        
        # Generate batch ID
        batch_id = str(uuid.uuid4())
        
        return BatchPredictionResponse(
            predictions=predictions,
            batch_id=batch_id,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Error in batch prediction endpoint: {str(e)}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                detail=str(e),
                timestamp=datetime.now(),
                error_code="BATCH_PREDICTION_ERROR"
            ).dict()
        )

@app.get("/predictions/{prediction_id}")
async def get_prediction(prediction_id: str, api_key: str = Depends(verify_api_key)):
    """Retrieve a stored prediction by ID."""
    try:
        # In the local version, we don't have a direct way to look up by ID
        # We'll search through all cached files
        for filename in os.listdir(local_settings.CACHE_DIR):
            if filename.endswith('.json'):
                with open(os.path.join(local_settings.CACHE_DIR, filename), 'r') as f:
                    data = json.load(f)
                    if data.get('prediction_id') == prediction_id:
                        return data
        
        raise HTTPException(status_code=404, detail="Prediction not found")
    except Exception as e:
        logger.error(f"Error retrieving prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
