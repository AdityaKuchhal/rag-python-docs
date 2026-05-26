# Lambda container image — Python 3.11 on ARM64 (cheaper + faster than x86)
FROM public.ecr.aws/lambda/python:3.11-arm64

# Copy requirements and install dependencies
COPY requirements-lambda.txt .
RUN pip install --no-cache-dir -r requirements-lambda.txt

# Copy application source
COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY pyproject.toml ${LAMBDA_TASK_ROOT}/

# Lambda handler — Mangum wraps FastAPI for Lambda compatibility
CMD ["src.api.lambda_handler.handler"]
