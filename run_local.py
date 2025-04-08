import uvicorn
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup the environment for local development."""
    # Create necessary directories
    base_dir = Path(__file__).resolve().parent
    os.makedirs(base_dir / "models", exist_ok=True)
    os.makedirs(base_dir / "logs", exist_ok=True)
    os.makedirs(base_dir / "cache", exist_ok=True)
    
    # Check if .env.local exists, if not create it
    env_file = base_dir / ".env.local"
    if not env_file.exists():
        logger.info("Creating .env.local file with default settings")
        with open(env_file, 'w') as f:
            f.write("""API_KEY=local_development_key
SECRET_KEY=local_development_secret
LOG_LEVEL=INFO
ENABLE_PROMETHEUS=False""")
    
    logger.info("Environment setup completed")

def main():
    """Run the local version of the application."""
    logger.info("Starting local version of Vehicle Predictive Maintenance API")
    
    # Setup environment
    setup_environment()
    
    # Run the application
    uvicorn.run(
        "app.api.local_endpoints:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )

if __name__ == "__main__":
    main()
