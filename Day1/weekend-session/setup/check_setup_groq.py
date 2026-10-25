"""Quick pre-flight check for the Groq editions: Groq API key and model access,
plus (optional) AWS credentials for the labs that still use AWS services.

Run:  python setup/check_setup_groq.py
"""

import os
import sys

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))  # pick up GROQ_API_KEY from a .env file
except ImportError:
    pass

from openai import OpenAI

MODELS = [
    os.getenv("GROQ_MODEL_ID", "openai/gpt-oss-120b"),  # default for every lab
    "openai/gpt-oss-20b",  # Mem0 fact extraction; faster/cheaper fallback
    "qwen/qwen3.8-27b",  # parallel tool calls (Section 2, async tools)
    "openai/gpt-oss-safeguard-20b",  # Section 7 guardrail
]

print(f"Python  : {sys.version.split()[0]}")
if sys.version_info < (3, 10):
    print("  ❌ Python 3.10+ is required (3.11 or 3.12 recommended)")

if not os.getenv("GROQ_API_KEY"):
    sys.exit("❌ GROQ_API_KEY is not set. Create one at https://console.groq.com/keys and export it (or put it in .env).")

base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")  # or a local OpenAI-compatible server
client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url=base_url, max_retries=3)
for model_id in MODELS:
    try:
        raw = client.chat.completions.with_raw_response.create(
            model=model_id,
            messages=[{"role": "user", "content": "Reply with the single word: ready"}],
            max_tokens=300,
        )
        text = (raw.parse().choices[0].message.content or "").strip()
        tpm = raw.headers.get("x-ratelimit-limit-tokens", "?")
        print(f"✅ {model_id}: {text[:20]}  (limit {tpm} tokens/minute)")
    except Exception as e:
        print(f"❌ {model_id}: {e}")

print(
    "\nFree-tier note: each model has its own limits (about 8,000 tokens/minute and 200,000 tokens/day for gpt-oss).\n"
    "If you hit the daily limit, set GROQ_MODEL_ID to another model, or upgrade to Groq's Dev tier."
)

# AWS is optional: Sections 1, 3 (notebook 1), 4, 5 and the guardrails notebook need only Groq.
try:
    import boto3

    region = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
    ident = boto3.client("sts", region_name=region).get_caller_identity()
    print(f"\nAWS     : account {ident['Account']} in {region} (for the labs that use AWS services)")
    try:
        boto3.client("bedrock-agentcore-control", region_name=region).list_agent_runtimes(maxResults=1)
        print("✅ AgentCore is reachable (Sections 3, 6 and 7 labs that use it)")
    except Exception as e:
        print(f"❌ AgentCore not reachable: {e}\n   (AWS Free-plan accounts can't use AgentCore; upgrade to the Paid plan)")
except ImportError:
    print("\nAWS     : boto3 not installed — skip unless you run the AWS-based labs")
except Exception as e:
    print(f"\nAWS     : no working credentials ({type(e).__name__}) — only needed for the AWS-based labs")

print("\nIf the Groq models replied, you're ready for Section 1 (Groq edition).")
