"""Quick pre-flight check: AWS credentials, region, and Bedrock model access.

Run:  python setup/check_setup.py
"""

import os
import sys

import boto3

REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
MODELS = [
    os.getenv("BEDROCK_MODEL_ID", "global.anthropic.claude-sonnet-4-6"),
    "global.anthropic.claude-haiku-4-5-20251001-v1:0",
]

print(f"Python  : {sys.version.split()[0]}")
if sys.version_info < (3, 10):
    print("  ❌ Python 3.10+ is required (3.11 or 3.12 recommended)")

try:
    ident = boto3.client("sts", region_name=REGION).get_caller_identity()
    print(f"Account : {ident['Account']}")
    print(f"Caller  : {ident['Arn']}")
except Exception as e:
    sys.exit(f"❌ AWS credentials not working: {e}")

print(f"Region  : {REGION}")

runtime = boto3.client("bedrock-runtime", region_name=REGION)
for model_id in MODELS:
    try:
        r = runtime.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": "Reply with the single word: ready"}]}],
            inferenceConfig={"maxTokens": 10},
        )
        text = r["output"]["message"]["content"][0]["text"].strip()
        print(f"✅ {model_id}: {text}")
    except Exception as e:
        print(f"❌ {model_id}: {e}")

print("\nIf both models replied, you're ready for Section 1.")
