"""
lambda_handler.py — Mangum adapter for AWS Lambda + API Gateway.

Mangum translates API Gateway HTTP events into ASGI requests
that FastAPI can handle. Zero changes needed to main.py.
"""

from mangum import Mangum
from src.api.main import app

# Configure Mangum for API Gateway HTTP API (payload v2)
handler = Mangum(app, lifespan="on")
