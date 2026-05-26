#!/bin/bash
set -e  # exit on any error

# ── Config ────────────────────────────────────────────────────
AWS_ACCOUNT_ID="975050138120"
AWS_REGION="us-east-1"
ECR_REPO="rag-python-docs"
LAMBDA_FUNCTION="rag-python-docs-api"
IMAGE_TAG="latest"

ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}"

echo "🔐 Authenticating with ECR..."
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin \
  "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo "🏗️  Building Docker image (arm64)..."
docker buildx build \
  --platform linux/arm64 \
  --provenance=false \
  -f infra/lambda/Dockerfile \
  -t "${ECR_URI}:${IMAGE_TAG}" \
  --push \
  .

echo "⚡ Updating Lambda function..."
aws lambda update-function-code \
  --function-name $LAMBDA_FUNCTION \
  --image-uri "${ECR_URI}:${IMAGE_TAG}" \
  --region $AWS_REGION

echo "⏳ Waiting for update to complete..."
aws lambda wait function-updated \
  --function-name $LAMBDA_FUNCTION \
  --region $AWS_REGION

echo "✅ Deployment complete!"
echo "🔗 Testing health endpoint..."
aws lambda invoke \
  --function-name $LAMBDA_FUNCTION \
  --region $AWS_REGION \
  --payload '{"version":"2.0","routeKey":"GET /health","rawPath":"/health","rawQueryString":"","headers":{"content-type":"application/json"},"requestContext":{"http":{"method":"GET","path":"/health"}}}' \
  --cli-binary-format raw-in-base64-out \
  /tmp/lambda_response.json && cat /tmp/lambda_response.json
