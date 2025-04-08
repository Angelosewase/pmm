from fastapi import FastAPI, HTTPException, Depends, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import logging
import json
from datetime import datetime
import aioredis
import hashlib
import asyncio
from app.core.config import settings
from app.preprocessing.data_processor import DataProcessor
from app.models.ml_model import MaintenanceModel
import numpy as np
from prometheus_client import Counter, Histogram
import time

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTION_COUNTER = Counter('prediction_requests_total', 'Total number of prediction requests')
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Prediction request latency')
ERROR_COUNTER = Counter('error_total', 'Total number of errors')

# Initialize FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Vehicle Predictive Maintenance API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Redis
redis = None

@app.on_event("startup")
async def startup_event():
    global redis
    redis = await aioredis.create_redis_pool(settings.REDIS_URL)
    logger.info("Connected to Redis")

@app.on_event("shutdown")
async def shutdown_event():
    if redis is not None:
        redis.close()
        await redis.wait_closed()
        logger.info("Disconnected from Redis")

# Initialize components
data_processor = DataProcessor(settings.DATA_PATH)
model = MaintenanceModel(settings.MODEL_PATH)

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
    if x_api_key != settings.API_KEY:
        ERROR_COUNTER.inc()
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    return x_api_key

def generate_cache_key(input_data: Dict[str, Any]) -> str:
    """Generate a cache key from input data."""
    sorted_items = sorted(input_data.items())
    return hashlib.md5(json.dumps(sorted_items).encode()).hexdigest()

async def store_prediction(prediction_data: Dict[str, Any]):
    """Store prediction data in Redis."""
    prediction_id = prediction_data['prediction_id']
    await redis.set(
        f"prediction:{prediction_id}",
        json.dumps(prediction_data),
        expire=settings.PREDICTION_CACHE_TTL
    )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        if redis is not None:
            await redis.ping()
            redis_status = "healthy"
        else:
            redis_status = "unavailable"
            
        return {
            "status": "healthy",
            "redis": redis_status,
            "timestamp": datetime.now()
        }
    except Exception as e:
        ERROR_COUNTER.inc()
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "detail": str(e)}
        )

@app.post("/train", dependencies=[Depends(verify_api_key)])
async def train_model(background_tasks: BackgroundTasks):
    """Train the model with current data."""
    try:
        # Load and preprocess data
        df = data_processor.load_data()
        df_processed = data_processor.preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = data_processor.split_data(df_processed)
        
        # Train model in background
        def train_model_task():
            try:
                model.train(X_train, y_train)
                metrics = model.evaluate(X_test, y_test)
                logger.info(f"Model training completed with metrics: {metrics}")
            except Exception as e:
                ERROR_COUNTER.inc()
                logger.error(f"Background training failed: {str(e)}")
        
        background_tasks.add_task(train_model_task)
        
        return {
            "status": "accepted",
            "message": "Model training started in background",
            "timestamp": datetime.now()
        }
    except Exception as e:
        ERROR_COUNTER.inc()
        logger.error(f"Error in training endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error training model: {str(e)}"
        )

@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
async def predict(input_data: PredictionInput):
    """Make predictions for new data."""
    PREDICTION_COUNTER.inc()
    start_time = time.time()
    
    try:
        # Generate cache key
        cache_key = generate_cache_key(input_data.dict())
        
        # Check cache
        cached_result = await redis.get(cache_key)
        if cached_result:
            return PredictionResponse(**json.loads(cached_result))
        
        # Convert input to dictionary
        input_dict = input_data.dict()
        
        # Prepare data for prediction
        X = data_processor.prepare_prediction_data(input_dict)
        
        # Make prediction
        predictions, probabilities = model.predict(X)
        
        # Get prediction and probability
        prediction = int(predictions[0])
        probability = float(probabilities[0][1])
        
        # Get feature importance
        feature_importance = model.feature_importance()
        
        # Generate prediction ID
        prediction_id = hashlib.md5(
            f"{cache_key}:{datetime.now().isoformat()}".encode()
        ).hexdigest()
        
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
        await redis.set(
            cache_key,
            json.dumps(response.dict()),
            expire=settings.PREDICTION_CACHE_TTL
        )
        
        # Store prediction data
        await store_prediction(response.dict())
        
        # Record latency
        PREDICTION_LATENCY.observe(time.time() - start_time)
        
        return response
        
    except Exception as e:
        ERROR_COUNTER.inc()
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
async def batch_predict(input_data: BatchPredictionInput, background_tasks: BackgroundTasks):
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
            "maintenance_required_percentage": (maintenance_required / total_predictions) * 100,
            "average_probability": total_probability / total_predictions
        }
        
        # Generate batch ID
        batch_id = hashlib.md5(
            f"batch:{datetime.now().isoformat()}".encode()
        ).hexdigest()
        
        return BatchPredictionResponse(
            predictions=predictions,
            batch_id=batch_id,
            summary=summary
        )
        
    except Exception as e:
        ERROR_COUNTER.inc()
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
        prediction_data = await redis.get(f"prediction:{prediction_id}")
        if prediction_data:
            return json.loads(prediction_data)
        else:
            raise HTTPException(status_code=404, detail="Prediction not found")
    except Exception as e:
        ERROR_COUNTER.inc()
        logger.error(f"Error retrieving prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 