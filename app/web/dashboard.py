import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import requests
import json
from app.core.config import settings

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Vehicle Predictive Maintenance Dashboard", className="text-center my-4")
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
                            dbc.Input(id="engine-rpm", type="number", value=1000)
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Lub Oil Pressure"),
                            dbc.Input(id="lub-oil-pressure", type="number", value=2.5)
                        ], width=6)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Fuel Pressure"),
                            dbc.Input(id="fuel-pressure", type="number", value=10.0)
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Coolant Pressure"),
                            dbc.Input(id="coolant-pressure", type="number", value=2.0)
                        ], width=6)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Lub Oil Temperature"),
                            dbc.Input(id="lub-oil-temp", type="number", value=80.0)
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Coolant Temperature"),
                            dbc.Input(id="coolant-temp", type="number", value=85.0)
                        ], width=6)
                    ]),
                    dbc.Button("Predict", id="predict-button", color="primary", className="mt-3")
                ])
            ])
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Prediction Results"),
                dbc.CardBody([
                    html.Div(id="prediction-result"),
                    dcc.Graph(id="probability-gauge")
                ])
            ])
        ], width=6)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Model Training"),
                dbc.CardBody([
                    dbc.Button("Retrain Model", id="train-button", color="success", className="mb-3"),
                    html.Div(id="training-metrics")
                ])
            ])
        ])
    ], className="mt-4")
])

@app.callback(
    [Output("prediction-result", "children"),
     Output("probability-gauge", "figure")],
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
        return "No prediction yet", {}
    
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
            headers={"X-API-Key": settings.API_KEY}
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Create gauge chart
            fig = px.pie(
                values=[result["probability"], 1 - result["probability"]],
                names=["Maintenance Required", "Normal"],
                hole=0.7,
                color_discrete_sequence=["red", "green"]
            )
            
            # Update layout
            fig.update_layout(
                showlegend=False,
                annotations=[dict(text=f"{result['probability']:.1%}", x=0.5, y=0.5, font_size=20, showarrow=False)]
            )
            
            # Create result message
            result_message = html.Div([
                html.H4(f"Status: {result['status']}", className="mb-3"),
                html.P(f"Probability: {result['probability']:.1%}"),
                html.P(f"Prediction: {'Maintenance Required' if result['prediction'] == 1 else 'Normal'}")
            ])
            
            return result_message, fig
            
        else:
            return f"Error: {response.text}", {}
            
    except Exception as e:
        return f"Error: {str(e)}", {}

@app.callback(
    Output("training-metrics", "children"),
    [Input("train-button", "n_clicks")]
)
def train_model(n_clicks):
    if n_clicks is None:
        return "No training performed yet"
    
    try:
        # Make API request to train model
        response = requests.post(
            "http://localhost:8000/train",
            headers={"X-API-Key": settings.API_KEY}
        )
        
        if response.status_code == 200:
            metrics = response.json()["metrics"]
            return html.Div([
                html.H4("Training Metrics", className="mb-3"),
                html.P(f"Accuracy: {metrics['accuracy']:.3f}"),
                html.P(f"Precision: {metrics['precision']:.3f}"),
                html.P(f"Recall: {metrics['recall']:.3f}"),
                html.P(f"F1 Score: {metrics['f1']:.3f}")
            ])
        else:
            return f"Error: {response.text}"
            
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run_server(debug=True, port=8050) 