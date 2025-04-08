# Vehicle Predictive Maintenance System

A comprehensive system for predicting vehicle maintenance needs using machine learning.

## Features

- Data preprocessing and feature engineering
- Machine learning model training (Random Forest/XGBoost)
- REST API with FastAPI
- Web interface for predictions
- Docker support
- Authentication and security
- Comprehensive logging and error handling

## Project Structure

```
.
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── endpoints.py
│   │   └── auth.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── security.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── ml_model.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── data_processor.py
│   └── web/
│       ├── __init__.py
│       └── dashboard.py
├── data/
│   └── vehicle_maintenance_data.csv
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   └── test_model.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Setup Instructions

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and update the values
5. Run the application:
   ```bash
   uvicorn app.api.endpoints:app --reload
   ```

## Docker Setup

Build and run using Docker:

```bash
docker-compose up --build
```

## API Endpoints

- `POST /train`: Retrain the model
- `POST /predict`: Make predictions
- `GET /health`: Check server status

## Authentication

The API uses API key authentication. Include your API key in the request header:
```
X-API-Key: your_api_key_here
```

## Testing

Run tests using pytest:
```bash
pytest tests/
```

## Example Usage

1. Train the model:
```bash
curl -X POST "http://localhost:8000/train" \
     -H "X-API-Key: your_api_key_here"
```

2. Make predictions:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "X-API-Key: your_api_key_here" \
     -H "Content-Type: application/json" \
     -d '{"engine_temp": 85, "rpm": 2000, "vibration": 0.5, "oil_pressure": 2.5, "mileage": 50000}'
```

## License

MIT License 