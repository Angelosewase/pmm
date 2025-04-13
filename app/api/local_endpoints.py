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

# Custom JSON encoder to handle datetime objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

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

class TrainingResponse(BaseModel):
    message: str
    status: str
    timestamp: datetime

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

def train_model_task():
    """Background task to train the model."""
    try:
        # Initialize components
        data_processor = DataProcessor(local_settings.DATA_PATH)
        model = MaintenanceModel(local_settings.MODEL_PATH)
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        df = data_processor.load_data()
        if df is None or df.empty:
            raise ValueError("Failed to load training data")
            
        # Verify required columns
        required_columns = data_processor.feature_columns + [data_processor.target_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Check for sufficient data
        if len(df) < 100:  # Minimum required samples
            raise ValueError(f"Insufficient training data: {len(df)} samples (minimum 100 required)")
            
        # Check class balance
        class_counts = df[data_processor.target_column].value_counts()
        min_class_count = class_counts.min()
        if min_class_count < 10:  # Minimum samples per class
            raise ValueError(f"Insufficient samples for class {class_counts.idxmin()}: {min_class_count} (minimum 10 required)")
            
        # Preprocess data
        processed_df = data_processor.preprocess_data(df)
        if processed_df is None:
            raise ValueError("Data preprocessing failed")
            
        # Split data
        X_train, X_test, y_train, y_test = data_processor.split_data(processed_df)
        
        # Train model
        logger.info("Training model...")
        model.train(X_train, y_train)
        
        # Evaluate model
        metrics = model.evaluate(X_test, y_test)
        
        # Validate metrics
        if metrics['accuracy'] < 0.5:
            logger.warning(f"Model performance below random chance (accuracy: {metrics['accuracy']:.3f})")
        
        if metrics['roc_auc'] < 0.5:
            logger.warning(f"Model ROC AUC below random chance (ROC AUC: {metrics['roc_auc']:.3f})")
            
        # Save feature importance
        feature_importance = model.feature_importance()
        if feature_importance:
            logger.info(f"Top features by importance: {dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5])}")
        
        return metrics
        
    except ValueError as ve:
        logger.error(f"Validation error in model training: {str(ve)}")
        raise
    except Exception as e:
        logger.error(f"Error in model training task: {str(e)}")
        raise

@app.post("/train", response_model=TrainingResponse, dependencies=[Depends(verify_api_key)])
async def train_model(background_tasks: BackgroundTasks) -> TrainingResponse:
    """Train the maintenance prediction model."""
    try:
        # Add training task to background tasks
        background_tasks.add_task(train_model_task)
        
        return TrainingResponse(
            message="Model training started in background",
            status="success",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error initiating model training: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start model training: {str(e)}"
        )

@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
async def predict(input_data: PredictionInput) -> PredictionResponse:
    """Make predictions for input data."""
    try:
        # Convert input data to dictionary and validate
        data_dict = input_data.dict()
        required_fields = {
            'engine_rpm', 'lub_oil_pressure', 'fuel_pressure',
            'coolant_pressure', 'lub_oil_temp', 'coolant_temp'
        }
        
        missing_fields = required_fields - set(data_dict.keys())
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Validate value ranges
        valid_ranges = {
            'engine_rpm': (300, 2000),
            'lub_oil_pressure': (2.0, 6.0),
            'fuel_pressure': (3.0, 20.0),
            'coolant_pressure': (1.0, 4.0),
            'lub_oil_temp': (70.0, 90.0),
            'coolant_temp': (70.0, 90.0)
        }
        
        for field, (min_val, max_val) in valid_ranges.items():
            value = data_dict.get(field)
            if value is not None and not (min_val <= float(value) <= max_val):
                logger.warning(f"{field} value {value} outside valid range [{min_val}, {max_val}]")
        
        # Initialize components
        data_processor = DataProcessor(local_settings.DATA_PATH)
        model = MaintenanceModel(local_settings.MODEL_PATH)
        
        try:
            # Prepare input data
            processed_data = data_processor.prepare_prediction_data(data_dict)
            
            # Make prediction
            predictions, probabilities = model.predict(processed_data)
            
            # Get prediction and probability
            prediction = int(predictions[0])
            probability = float(probabilities[0][1])  # Probability of positive class
            
            # Get feature importance
            feature_importance = model.feature_importance()
            
            # Generate prediction ID
            prediction_id = hashlib.md5(
                f"{datetime.now().isoformat()}:{str(input_data.dict())}".encode()
            ).hexdigest()

            # Create response with all required fields
            response = PredictionResponse(
                prediction=prediction,
                probability=probability,
                status="Maintenance Required" if prediction == 1 else "Normal",
                feature_importance=feature_importance,
                prediction_id=prediction_id,
                timestamp=datetime.now()
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in prediction processing: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing prediction: {str(e)}"
            )
            
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise HTTPException(
            status_code=400,
            detail=str(ve)
        )
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
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
            content=json.dumps(ErrorResponse(
                detail=str(e),
                timestamp=datetime.now(),
                error_code="BATCH_PREDICTION_ERROR"
            ).dict(), cls=CustomJSONEncoder)
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
