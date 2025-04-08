from pydantic import BaseSettings
from typing import List
import os
from pathlib import Path

class Settings(BaseSettings):
    # Project settings
    PROJECT_NAME: str = "Vehicle Predictive Maintenance API"
    VERSION: str = "1.0.0"
    API_KEY: str = os.getenv("API_KEY", "your-api-key-here")
    
    # Path settings
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_PATH: str = str(BASE_DIR / "data" / "vehicle_maintenance_data.csv")
    MODEL_PATH: str = str(BASE_DIR / "models" / "maintenance_model.joblib")
    LOG_PATH: str = str(BASE_DIR / "logs")
    
    # API settings
    ALLOWED_ORIGINS: List[str] = ["*"]  # Update in production
    API_V1_STR: str = "/api/v1"
    
    # Redis settings
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB: int = int(os.getenv("REDIS_DB", 0))
    REDIS_URL: str = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
    PREDICTION_CACHE_TTL: int = 3600  # 1 hour
    
    # Model settings
    MODEL_UPDATE_FREQUENCY: int = 24  # hours
    MIN_TRAINING_SAMPLES: int = 1000
    FEATURE_SELECTION_THRESHOLD: float = 0.05
    
    # Monitoring settings
    ENABLE_PROMETHEUS: bool = True
    PROMETHEUS_PORT: int = 9090
    LOG_LEVEL: str = "INFO"
    
    # Security settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Performance settings
    MAX_WORKERS: int = 4
    BATCH_SIZE: int = 100
    REQUEST_TIMEOUT: int = 30
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()

# Create necessary directories
os.makedirs(settings.LOG_PATH, exist_ok=True)
os.makedirs(os.path.dirname(settings.MODEL_PATH), exist_ok=True) 