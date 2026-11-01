# Deploy a Strands Agent to AWS Lambda

A serverless contract-analysis agent behind API Gateway. POST a document; get back key terms, flagged risks and a summary.

```
User → API Gateway → Lambda (container image) → Strands agent → Amazon Bedrock
```

## Prerequisites

1. AWS credentials **in the terminal you deploy from** (`aws sts get-caller-identity` must print your account). With SSO: `aws sso login --profile <p>`, then `set -a; source ../../.env; set +a` (or `export AWS_PROFILE=<p>`)
2. AWS SAM CLI (`sam --version`). If `deploy.sh` says `ERROR: AWS SAM CLI is not installed.`, install it: `brew install aws-sam-cli` (macOS), or in its own environment with `uv tool install aws-sam-cli` or `pipx install aws-sam-cli`. Avoid installing it into the course venv — it pins many packages
3. Docker running (`docker info`)
4. A model you can call: if Claude isn't enabled in your account yet, `export BEDROCK_MODEL_ID=us.amazon.nova-2-lite-v1:0` (see "Before you start" in the section README)

On Windows, run the scripts from WSL (recommended) or Git Bash.

## Deploy

```bash
./deploy.sh          # or: bash deploy.sh
```

The script detects your CPU architecture (arm64 for Apple Silicon/Graviton, x86_64 otherwise), builds the image, and deploys with SAM. Set `BEDROCK_MODEL_ID` before running to use a different model; running `./deploy.sh` again with a new value updates the function in about a minute.

- **How long:** the script says 2–3 minutes, but the first deploy pushes a ~1 GB image to ECR and typically takes **5–8 minutes**. Later deploys take about a minute.
- **What it creates:** the stack `document-analysis-agent` (Lambda function, IAM role, API Gateway, an S3 bucket) and a companion stack with the ECR repository. `sam deploy --resolve-s3` also creates SAM's own stack, `aws-sam-cli-managed-default`, the first time you use SAM in a region: a small S3 bucket where SAM uploads templates, shared by every SAM app in that region.

## Test

```bash
curl -X POST <ApiEndpoint printed by deploy.sh> \
  -H "Content-Type: application/json" \
  -d '{"document_text": "This agreement is entered into by Party A and Party B. Party A shall deliver 1000 units by March 2027. Late delivery incurs unlimited liability."}'
```

A good answer has sections for the parties and key terms, the risky clauses (quoted) and a summary. The first request is slower (cold start, 5–15 s); warm requests take about 4 s.

What the status codes mean:

| Response | Meaning |
|---|---|
| `200` | The agent's analysis |
| `400 {"error": "Missing 'document_text' in request body"}` | The body was `{}`, empty, or used another key |
| `502 {"message": "Internal server error"}` | The function raised an error. Common causes: the body isn't valid JSON or isn't a JSON object; Bedrock refused the call (expired model access, the Anthropic use-case form not submitted); or the model ran out of output tokens on a long answer (`MaxTokensReachedException`, seen occasionally with Nova 2 Lite — just retry). The reason is in the function's log |
| `504` | Took longer than API Gateway's 29 s limit |

Read the function's log:

```bash
LOG_GROUP=$(aws logs describe-log-groups --log-group-name-prefix /aws/lambda/document-analysis-agent \
  --query 'logGroups[0].logGroupName' --output text)
aws logs tail "$LOG_GROUP" --since 15m
```

> API Gateway REST APIs time out after 29 seconds by default. Keep requests short, use a faster model (`global.anthropic.claude-haiku-4-5-20251001-v1:0`), or use a Lambda Function URL / AgentCore Runtime for long-running agents.

> ⚠️ The demo API has no authentication — anyone with the URL can call it and spend your Bedrock budget. Don't share the URL, and run `./cleanup.sh` the same day. For real use, add an API key/usage plan, IAM auth or a Cognito authorizer.

## Clean up

```bash
./cleanup.sh
```

It deletes the stack and the ECR repository (about a minute). It prints "All resources removed", but two things stay:

- **The function's log group** (`/aws/lambda/document-analysis-agent-...`): `aws logs delete-log-group --log-group-name "$LOG_GROUP"` (see Test above for `LOG_GROUP`).
- **SAM's `aws-sam-cli-managed-default` stack and bucket.** Keep it if you use SAM for anything else in this region. To remove it: empty the bucket in the S3 console (the "Empty" button also removes old versions — the bucket is versioned), then `aws cloudformation delete-stack --stack-name aws-sam-cli-managed-default`.

`sam delete` may also warn `Cannot resolve s3 bucket information`; that's because `deploy.sh` doesn't save a `samconfig.toml`. The stack is still deleted.

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
