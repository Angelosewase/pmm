from app.web.local_dashboard import app

if __name__ == "__main__":
    print("Starting Vehicle Predictive Maintenance Dashboard...")
    print("Open your browser at http://127.0.0.1:8050")
    app.run_server(debug=True, port=8050)
