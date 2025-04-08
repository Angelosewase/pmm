#!/bin/bash
set -e

# Wait for Redis to be ready
echo "Waiting for Redis..."
while ! nc -z ${REDIS_HOST:-redis} ${REDIS_PORT:-6379}; do
  sleep 1
done
echo "Redis is ready!"

# Start Prometheus metrics exporter
python -m prometheus_client.exposition --port=9090 &

# Start the FastAPI application with uvicorn
exec uvicorn app.api.endpoints:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers ${MAX_WORKERS:-4} \
    --log-level ${LOG_LEVEL:-info} 