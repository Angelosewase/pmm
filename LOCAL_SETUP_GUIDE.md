# Vehicle Predictive Maintenance System - Local Setup Guide

This guide will help you set up and run the Vehicle Predictive Maintenance System locally without using Docker.

## Improvements Made

I've made several improvements to the original project:

1. **Simplified Local Setup**: Created a version that doesn't require Redis or other external dependencies
2. **File-based Caching**: Implemented a simple file-based caching system to replace Redis
3. **Enhanced Dashboard**: Improved the web dashboard with better visualizations and error handling
4. **Sample Data Generator**: Added a script to generate sample data for testing
5. **Comprehensive Documentation**: Created this detailed guide for local setup and usage
6. **Streamlined Configuration**: Created a dedicated local configuration with sensible defaults

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

## Setup Instructions

### 1. Create and Activate a Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
# source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install the local requirements
pip install -r requirements-local.txt
```

### 3. Generate Sample Data (if needed)

If you don't have real vehicle maintenance data or want to test with synthetic data:

```bash
python scripts\generate_sample_data.py
```

This will create a sample dataset with 5,000 records in `data\vehicle_maintenance_data.csv`.

## Running the Application

### 1. Start the API Server

```bash
python run_local.py
```

This will start the FastAPI server at http://127.0.0.1:8000

You can access the API documentation at:
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

### 2. Start the Dashboard (in a separate terminal)

```bash
# Make sure your virtual environment is activated
python run_dashboard.py
```

This will start the dashboard at http://127.0.0.1:8050

## Using the System

### API Endpoints

The following API endpoints are available:

1. **Health Check**
   - `GET /health`
   - Check if the API and model are working correctly

2. **Train Model**
   - `POST /train`
   - Train or retrain the model with the latest data
   - Requires API key in header: `X-API-Key: local_development_key`

3. **Make Prediction**
   - `POST /predict`
   - Make a prediction for a single set of vehicle parameters
   - Requires API key in header: `X-API-Key: local_development_key`
   - Example request body:
     ```json
     {
       "engine_rpm": 1800,
       "lub_oil_pressure": 2.5,
       "fuel_pressure": 3.0,
       "coolant_pressure": 1.2,
       "lub_oil_temp": 85.0,
       "coolant_temp": 80.0
     }
     ```

4. **Batch Prediction**
   - `POST /predict/batch`
   - Make predictions for multiple sets of parameters
   - Requires API key in header: `X-API-Key: local_development_key`

### Dashboard

The dashboard provides a user-friendly interface to:

1. Input vehicle parameters and make predictions
2. View prediction results with probability gauge
3. Visualize feature importance
4. Train/retrain the model
5. Check API status

## How It Works

### Architecture

The system consists of several components:

1. **Data Processor** (`app/preprocessing/data_processor.py`)
   - Handles data loading, cleaning, and preprocessing
   - Applies feature engineering and selection
   - Prepares data for model training and prediction

2. **Machine Learning Model** (`app/models/ml_model.py`)
   - Implements an XGBoost classifier with Optuna hyperparameter optimization
   - Includes model training, evaluation, and prediction functionality
   - Provides feature importance analysis

3. **API** (`app/api/local_endpoints.py`)
   - Exposes REST endpoints for prediction and model training
   - Handles request validation and error handling
   - Implements a simple file-based caching system

4. **Dashboard** (`app/web/local_dashboard.py`)
   - Provides a web interface built with Dash and Plotly
   - Visualizes predictions and model metrics
   - Allows for interactive parameter input and model training

### Prediction Process

1. User inputs vehicle parameters (via API or dashboard)
2. Data is preprocessed and normalized
3. Model makes a prediction (0 = normal, 1 = maintenance required)
4. System returns prediction with probability and feature importance
5. Results are cached for future identical requests

### Training Process

1. System loads and preprocesses the training data
2. Hyperparameter optimization is performed using Optuna
3. Model is trained with the best parameters
4. Performance metrics are calculated and returned
5. Model is saved for future predictions

## Customization

### Configuration

You can modify the local configuration in `app/core/local_config.py` to adjust:

- API settings
- Model parameters
- File paths
- Caching behavior

### Environment Variables

The system uses environment variables from `.env.local` for configuration. You can modify this file to change:

- API key
- Secret key
- Log level

## Troubleshooting

### Common Issues

1. **API Connection Error**
   - Make sure the API server is running on port 8000
   - Check if there are any error messages in the API server terminal

2. **Model Training Fails**
   - Ensure you have sufficient data in the data file
   - Check if the data format matches the expected format

3. **Dashboard Doesn't Show Predictions**
   - Verify that the API key in the dashboard matches the one in the API server
   - Check the browser console for any JavaScript errors

### Logs

Logs are stored in the `logs` directory and can be helpful for debugging issues.

## Next Steps

To further improve the system, consider:

1. Adding more advanced feature engineering
2. Implementing A/B testing for model comparison
3. Adding user authentication to the dashboard
4. Implementing anomaly detection for early warning
5. Adding time series forecasting for predictive maintenance scheduling

## Support

If you encounter any issues or have questions, please open an issue in the project repository.
