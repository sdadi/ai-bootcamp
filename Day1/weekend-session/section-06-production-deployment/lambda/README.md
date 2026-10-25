# Deploy a Strands Agent to AWS Lambda

A serverless contract-analysis agent behind API Gateway. POST a document; get back key terms, flagged risks and a summary.

```
User → API Gateway → Lambda (container image) → Strands agent → Amazon Bedrock
```

## Prerequisites

1. AWS CLI configured (`aws sts get-caller-identity`)
2. AWS SAM CLI (`sam --version`)
3. Docker running (`docker info`)

On Windows, run the scripts from WSL (recommended) or Git Bash.

## Deploy

```bash
chmod +x deploy.sh cleanup.sh
./deploy.sh
```

The script detects your CPU architecture (arm64 for Apple Silicon/Graviton, x86_64 otherwise), builds the image, and deploys with SAM. Takes 2–3 minutes. Set `BEDROCK_MODEL_ID` before running to use a different model.

## Test

```bash
curl -X POST <ApiEndpoint printed by deploy.sh> \
  -H "Content-Type: application/json" \
  -d '{"document_text": "This agreement is entered into by Party A and Party B. Party A shall deliver 1000 units by March 2027. Late delivery incurs unlimited liability."}'
```

The first request is slower (cold start, 5–15 s).

> API Gateway REST APIs time out after 29 seconds by default. Keep requests short, use a faster model (`global.anthropic.claude-haiku-4-5-20251001-v1:0`), or use a Lambda Function URL / AgentCore Runtime for long-running agents.

> ⚠️ The demo API has no authentication — anyone with the URL can call it and spend your Bedrock budget. Don't share the URL, and run `./cleanup.sh` the same day. For real use, add an API key/usage plan, IAM auth or a Cognito authorizer.

## Clean up

```bash
./cleanup.sh
```

## Groq edition (no Bedrock models)

| File | Difference from the Bedrock version |
|---|---|
| `src/app_groq.py`, `src/requirements-groq.txt`, `src/Dockerfile.groq` | The model runs on Groq (`openai/gpt-oss-120b` by default) |
| `template-groq.yaml` | No Bedrock IAM policy; `GROQ_API_KEY` comes from Secrets Manager through a `{{resolve:secretsmanager:...}}` reference, so the key is never in the template, the stack parameters or the deploy command |
| `deploy-groq.sh` / `cleanup-groq.sh` | Store the key in the secret `document-analysis-agent-groq/groq-api-key` first, then build and deploy stack `document-analysis-agent-groq`; cleanup deletes the stack, then the secret |

```bash
export GROQ_API_KEY=...            # or: set -a; source ../../.env; set +a
./deploy-groq.sh                   # ./cleanup-groq.sh when done
```

- **SAM without Homebrew:** `pip install aws-sam-cli` in a virtual environment works too.
- **Timings:** the first deploy pushes a ~1 GB image to ECR — minutes on a fast line, much longer on a slow one. The first request after a deploy can take ~25 s (the first init times out while the image loads, then re-runs), close to API Gateway's 29 s limit: retry once if you get a 504. Later cold starts are ~5 s, warm requests ~1–2 s.
- **Cleanup order matters:** CloudFormation reads the secret even while deleting the stack, so `cleanup-groq.sh` deletes the stack first. If the secret is gone first, recreate it with any value and run `./cleanup-groq.sh` again.
- **Changing the key:** re-running `deploy-groq.sh` with only a new key updates the secret but not the function (the changeset is empty). Delete and redeploy, or change something else in the stack.
- The resolved key is visible in the Lambda console's environment variables to anyone with `lambda:GetFunctionConfiguration`; for production, read it from Secrets Manager at runtime instead.
