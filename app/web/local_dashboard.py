import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import requests
import json
from app.core.local_config import local_settings

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # For production deployment

# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Vehicle Predictive Maintenance Dashboard", className="text-center my-4"),
            html.P("Local Development Version", className="text-center text-muted")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Input Parameters"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Engine RPM"),
                            dbc.Input(id="engine-rpm", type="number", value=1800, min=0, max=5000)
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Lub Oil Pressure"),
                            dbc.Input(id="lub-oil-pressure", type="number", value=2.5, min=0, max=10, step=0.1)
                        ], width=6)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Fuel Pressure"),
                            dbc.Input(id="fuel-pressure", type="number", value=3.0, min=0, max=10, step=0.1)
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Coolant Pressure"),
                            dbc.Input(id="coolant-pressure", type="number", value=1.2, min=0, max=5, step=0.1)
                        ], width=6)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Lub Oil Temperature"),
                            dbc.Input(id="lub-oil-temp", type="number", value=85.0, min=0, max=150)
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Coolant Temperature"),
                            dbc.Input(id="coolant-temp", type="number", value=80.0, min=0, max=150)
                        ], width=6)
                    ]),
                    dbc.Button("Predict", id="predict-button", color="primary", className="mt-3 w-100")
                ])
            ])
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Prediction Results"),
                dbc.CardBody([
                    html.Div(id="prediction-result", className="mb-4"),
                    dcc.Graph(id="probability-gauge", config={'displayModeBar': False})
                ])
            ])
        ], width=6)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Feature Importance"),
                dbc.CardBody([
                    dcc.Graph(id="feature-importance-chart", config={'displayModeBar': False})
                ])
            ])
        ], width=12)
    ], className="mt-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Model Training"),
                dbc.CardBody([
                    html.P("Train or retrain the model with the latest data:"),
                    dbc.Button("Retrain Model", id="train-button", color="success", className="mb-3 w-100"),
                    html.Div(id="training-status", className="mt-3")
                ])
            ])
        ], width=12)
    ], className="mt-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("API Status"),
                dbc.CardBody([
                    dbc.Button("Check API Status", id="status-button", color="info", className="mb-3"),
                    html.Div(id="api-status")
                ])
            ])
        ], width=12)
    ], className="mt-4"),
    
    html.Footer([
        html.P("Vehicle Predictive Maintenance System - Local Version", className="text-center text-muted mt-4")
    ])
], fluid=True)

@app.callback(
    [Output("prediction-result", "children"),
     Output("probability-gauge", "figure"),
     Output("feature-importance-chart", "figure")],
    [Input("predict-button", "n_clicks")],
    [State("engine-rpm", "value"),
     State("lub-oil-pressure", "value"),
     State("fuel-pressure", "value"),
     State("coolant-pressure", "value"),
     State("lub-oil-temp", "value"),
     State("coolant-temp", "value")]
)
def update_prediction(n_clicks, engine_rpm, lub_oil_pressure, fuel_pressure,
                     coolant_pressure, lub_oil_temp, coolant_temp):
    if n_clicks is None:
        # Default empty figures
        empty_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=0,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [0, 1]}}
        ))
        empty_gauge.update_layout(height=250)
        
        empty_bar = go.Figure()
        empty_bar.update_layout(height=300)
        
        return "No prediction yet", empty_gauge, empty_bar
    
    try:
        # Prepare input data
        input_data = {
            "engine_rpm": engine_rpm,
            "lub_oil_pressure": lub_oil_pressure,
            "fuel_pressure": fuel_pressure,
            "coolant_pressure": coolant_pressure,
            "lub_oil_temp": lub_oil_temp,
            "coolant_temp": coolant_temp
        }
        
        # Make API request
        response = requests.post(
            "http://localhost:8000/predict",
            json=input_data,
            headers={"X-API-Key": "your-secret-api-key"}
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Create gauge chart
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=result["probability"],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Failure Probability"},
                delta={'reference': 0.5},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "darkred" if result["probability"] > 0.5 else "green"},
                    'steps': [
                        {'range': [0, 0.3], 'color': "lightgreen"},
                        {'range': [0.3, 0.7], 'color': "lightyellow"},
                        {'range': [0.7, 1], 'color': "salmon"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.5
                    }
                }
            ))
            gauge_fig.update_layout(height=250)
            
            # Create feature importance chart
            feature_importance = result.get("feature_importance", {})
            if feature_importance:
                features = list(feature_importance.keys())
                values = list(feature_importance.values())
                
                # Sort by importance
                sorted_indices = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
                sorted_features = [features[i] for i in sorted_indices]
                sorted_values = [values[i] for i in sorted_indices]
                
                # Create bar chart
                importance_fig = go.Figure(go.Bar(
                    x=sorted_features,
                    y=sorted_values,
                    marker_color='royalblue'
                ))
                importance_fig.update_layout(
                    title="Feature Importance",
                    xaxis_title="Feature",
                    yaxis_title="Importance",
                    height=300
                )
            else:
                # Empty chart if no feature importance
                importance_fig = go.Figure()
                importance_fig.update_layout(
                    title="Feature Importance Not Available",
                    height=300
                )
            
            # Create result message
            status_color = "danger" if result["prediction"] == 1 else "success"
            result_message = html.Div([
                html.H4([
                    "Status: ",
                    html.Span(result["status"], className=f"text-{status_color}")
                ], className="mb-3"),
                html.P([
                    "Prediction: ",
                    html.Span(
                        "Maintenance Required" if result["prediction"] == 1 else "Normal",
                        className=f"text-{status_color} fw-bold"
                    )
                ]),
                html.P(f"Probability: {result['probability']:.2f}"),
                html.P(f"Prediction ID: {result['prediction_id']}")
            ])
            
            return result_message, gauge_fig, importance_fig
            
        else:
            error_message = html.Div([
                html.H4("Error", className="text-danger"),
                html.P(f"Status Code: {response.status_code}"),
                html.P(f"Message: {response.text}")
            ])
            
            empty_fig = go.Figure()
            empty_fig.update_layout(height=250)
            
            return error_message, empty_fig, empty_fig
            
    except Exception as e:
        error_message = html.Div([
            html.H4("Error", className="text-danger"),
            html.P(f"Message: {str(e)}")
        ])
        
        empty_fig = go.Figure()
        empty_fig.update_layout(height=250)
        
        return error_message, empty_fig, empty_fig

@app.callback(
    Output("training-status", "children"),
    [Input("train-button", "n_clicks")]
)
def train_model(n_clicks):
    if n_clicks is None:
        return "No training performed yet"
    
    try:
        # Make API request to train model
        response = requests.post(
            "http://localhost:8000/train",
            headers={"X-API-Key": local_settings.API_KEY}
        )
        
        if response.status_code == 200:
            result = response.json()
            return html.Div([
                html.H5("Training Status", className="text-success"),
                html.P(f"Status: {result.get('status', 'Unknown')}"),
                html.P(f"Message: {result.get('message', 'No message')}"),
                html.P(f"Timestamp: {result.get('timestamp', 'Unknown')}")
            ])
        else:
            return html.Div([
                html.H5("Training Error", className="text-danger"),
                html.P(f"Status Code: {response.status_code}"),
                html.P(f"Error: {response.text}")
            ])
            
    except Exception as e:
        return html.Div([
            html.H5("Training Error", className="text-danger"),
            html.P(f"Error: {str(e)}")
        ])

@app.callback(
    Output("api-status", "children"),
    [Input("status-button", "n_clicks")]
)
def check_api_status(n_clicks):
    if n_clicks is None:
        return ""
    
    try:
        # Make API request to check health
        response = requests.get("http://localhost:8000/health")
        
        if response.status_code == 200:
            status_data = response.json()
            return html.Div([
                html.H5("API Status", className="text-success"),
                html.P(f"Status: {status_data.get('status', 'Unknown')}"),
                html.P(f"Version: {status_data.get('version', 'Unknown')}"),
                html.P(f"Model Loaded: {'Yes' if status_data.get('model_loaded', False) else 'No'}"),
                html.P(f"Data Available: {'Yes' if status_data.get('data_available', False) else 'No'}"),
                html.P(f"Timestamp: {status_data.get('timestamp', 'Unknown')}")
            ])
        else:
            return html.Div([
                html.H5("API Status Error", className="text-danger"),
                html.P(f"Status Code: {response.status_code}"),
                html.P(f"Error: {response.text}")
            ])
            
    except Exception as e:
        return html.Div([
            html.H5("API Connection Error", className="text-danger"),
            html.P(f"Error: {str(e)}"),
            html.P("Make sure the API server is running on http://localhost:8000")
        ])

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
