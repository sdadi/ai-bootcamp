#!/bin/bash
set -e

STACK_NAME="document-analysis-agent-groq"
SECRET_NAME="${STACK_NAME}/groq-api-key"
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"

echo "=== Cleaning up Lambda deployment (Groq) ==="
echo ""

echo "Deleting stack '$STACK_NAME'..."
sam delete \
  --stack-name $STACK_NAME \
  --region $REGION \
  --no-prompts

echo "Deleting the Groq API key secret..."
aws secretsmanager delete-secret --secret-id "$SECRET_NAME" --force-delete-without-recovery \
  --region $REGION --output text > /dev/null 2>&1 || true

echo ""
echo "=== Cleanup complete. All resources removed. ==="
