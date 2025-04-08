from pydantic import BaseSettings
from typing import List
import os
from pathlib import Path

class LocalSettings(BaseSettings):
    # Project settings
    PROJECT_NAME: str = "Vehicle Predictive Maintenance API"
    VERSION: str = "1.0.0"
    API_KEY: str = os.getenv("API_KEY", "local_development_key")
    
    # Path settings
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_PATH: str = str(BASE_DIR / "data" / "vehicle_maintenance_data.csv")
    MODEL_PATH: str = str(BASE_DIR / "models" / "maintenance_model.joblib")
    LOG_PATH: str = str(BASE_DIR / "logs")
    
    # API settings
    ALLOWED_ORIGINS: List[str] = ["*"]
    API_V1_STR: str = "/api/v1"
    
    # Model settings
    MODEL_UPDATE_FREQUENCY: int = 24  # hours
    MIN_TRAINING_SAMPLES: int = 100  # Reduced for local testing
    FEATURE_SELECTION_THRESHOLD: float = 0.05
    
    # Monitoring settings
    ENABLE_PROMETHEUS: bool = False  # Disabled for local setup
    LOG_LEVEL: str = "INFO"
    
    # Security settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "local_development_secret")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Performance settings
    MAX_WORKERS: int = 2
    BATCH_SIZE: int = 50
    REQUEST_TIMEOUT: int = 30
    
    # Local file-based cache settings
    CACHE_DIR: str = str(BASE_DIR / "cache")
    PREDICTION_CACHE_TTL: int = 3600  # 1 hour
    
    class Config:
        case_sensitive = True
        env_file = ".env.local"

local_settings = LocalSettings()

# Create necessary directories
os.makedirs(local_settings.LOG_PATH, exist_ok=True)
os.makedirs(os.path.dirname(local_settings.MODEL_PATH), exist_ok=True)
os.makedirs(local_settings.CACHE_DIR, exist_ok=True)
