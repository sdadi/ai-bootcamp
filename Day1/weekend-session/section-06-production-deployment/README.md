# Section 6 — Shipping Agents to Production

| Folder | Target | Example |
|---|---|---|
| `agentcore-runtime/` | Amazon Bedrock AgentCore Runtime | Contract-analysis agent deployed with the starter toolkit (no Docker needed) |
| `ecs-fargate/` | Amazon ECS on Fargate behind an ALB | Hospital surgical-scheduling agent with a chat UI |
| `lambda/` | AWS Lambda behind API Gateway | Serverless contract-analysis agent |
| `agentcore-gateway/` | AgentCore Gateway | Turn a REST API (OpenAPI spec in S3) into MCP tools for an agent |
| `multi-agent-on-agentcore/` | AgentCore Runtime (MCP + A2A) | Finance orchestrator → stock agent (A2A) → stock tools (MCP) |
| `setup_cognito.sh` | Cognito | OAuth user pool + token for the multi-agent deployment (saves the pool in `.env.cognito` and reuses it on later runs) |

## Which target when?

| | Lambda | ECS Fargate | AgentCore Runtime |
|---|---|---|---|
| Shape | Function per request | Always-on web service | Managed agent runtime, one microVM per session |
| Max run time | 15 min (API Gateway: 29 s by default) | Unlimited | Up to 8 hours per session |
| You manage | Packaging | Containers, load balancer, scaling | Just the agent code |
| Good for | Short, bursty, event-driven tasks | Existing container platforms, custom web apps | Long-running, stateful, multi-user agents |

## Prerequisites

- AWS credentials and `AWS_REGION=us-east-1`
- `agentcore-runtime/`, `agentcore-gateway/`, `multi-agent-on-agentcore/`: `pip install bedrock-agentcore-starter-toolkit` (the multi-agent lab also needs `jq`)
- `ecs-fargate/`: Docker running, and a default VPC
- `lambda/`: Docker running and the AWS SAM CLI

## Before you start (5 minutes, saves an hour)

Every lab calls AWS and a Bedrock model, from three different places: notebooks, shell scripts, and the deployed agent. Set these up once:

1. **Log in to AWS, and put your settings where every tool finds them.** Run these from the repo root, replacing `<your-profile>` with your AWS CLI profile name.

   Log in (SSO sessions expire after some hours; when you see `ExpiredToken`, run this again):
   ```bash
   aws sso login --profile <your-profile>
   ```
   Add this section's settings to the repo's `.env` (notebooks opened in VS Code load it into the kernel), then open `.env` and set `AWS_PROFILE` to your profile:
   ```bash
   cat section-06-production-deployment/.env.example >> .env
   ```
   [`.env.example`](.env.example) explains each line: your AWS profile and region, `REGION` for the multi-agent lab, `AWS_PAGER=` (empty, so scripts never stop at `(END)`), a commented-out Nova model for step 2, and the Cognito test user.
   Shell scripts and `python agent.py` only see your terminal, not `.env`. Make the course venv load `.env` every time it's activated (once; safe to paste twice; on Windows Git Bash use `.venv/Scripts/activate`):
   ```bash
   grep -q '_course_env' .venv/bin/activate || cat >> .venv/bin/activate <<'EOF'

   # Load the repo's .env every time this venv is activated
   _course_env="$(dirname "$VIRTUAL_ENV")/.env"
   if [ -f "$_course_env" ]; then
       set -a; . "$_course_env"; set +a
   fi
   unset _course_env
   EOF
   source .venv/bin/activate
   aws sts get-caller-identity          # must print your account, not an error
   ```
   From now on every terminal that activates the venv has your profile and model; VS Code activates it in new terminals automatically. After editing `.env`, open a new terminal (or run `source .venv/bin/activate` again). If you ever recreate `.venv`, paste the block again. Without this step, run `set -a; source ../../.env; set +a` from a lab folder in each new terminal.

2. **Make sure the model works in your account.** The labs default to `global.anthropic.claude-sonnet-4-6` (Claude Sonnet 4.6). Anthropic models need a one-time **use-case form** per AWS account: Bedrock console → Model catalog → open any Claude model → submit the use-case details; access starts about 15 minutes later. Check it:
   ```bash
   aws bedrock-runtime converse --model-id global.anthropic.claude-sonnet-4-6 \
     --messages '[{"role":"user","content":[{"text":"Reply: pong"}]}]' \
     --query 'output.message.content[0].text' --output text
   ```
   If that fails with `Model use case details have not been submitted for this account`, use Amazon Nova until the form is approved. All five labs were tested end to end with it:
   ```bash
   export BEDROCK_MODEL_ID=us.amazon.nova-2-lite-v1:0
   ```
   To make it stick, uncomment the `BEDROCK_MODEL_ID` line in `.env` (step 1) and open a new terminal; restart the notebook kernel too.
   Every lab reads `BEDROCK_MODEL_ID`: the notebooks directly, the ECS and Lambda deploy scripts pass it into the deployment, and on AgentCore you pass it with `agentcore deploy --env BEDROCK_MODEL_ID=...` (see the multi-agent README).

3. **The AWS CLI pager is off** if you did step 1 (`AWS_PAGER=` in `.env`). Otherwise run `export AWS_PAGER=""` in the terminal you run scripts from — without it, long command output opens in a viewer, the script stops at `(END)`, and nothing happens until you press `q`.

4. **Pick the course virtual environment as the notebook kernel.** In VS Code: Select Kernel (top right) → Python Environments → `.venv`. If notebooks can't find files that sit next to them (e.g. `weather_openapi.json`), see Troubleshooting below.

## Quick start

```bash
# AgentCore Runtime
cd agentcore-runtime && jupyter lab deploy_to_agentcore_runtime.ipynb

# ECS Fargate
cd ecs-fargate && ./deploy.sh      # ./cleanup.sh when done

# Lambda
cd lambda && ./deploy.sh           # ./cleanup.sh when done

# AgentCore Gateway
cd agentcore-gateway && jupyter lab gateway_openapi_target.ipynb

# Multi-agent (local first, then AgentCore) — see multi-agent-on-agentcore/README.md
```

All the agents default to `global.anthropic.claude-sonnet-4-6`; set `BEDROCK_MODEL_ID` to change it (the deploy scripts pass it through).

| Lab | Takes (measured) | Its cleanup leaves behind |
|---|---|---|
| AgentCore Runtime | ~3 min | CloudWatch log groups and trace delivery settings — see its README |
| ECS Fargate | ~6 min deploy (first build on Apple Silicon is slower) + ~2 min until healthy; ~8 min cleanup | Deregistered (INACTIVE) task definitions |
| Lambda | ~5–8 min first deploy, ~1 min after | The Lambda log group and SAM's own `aws-sam-cli-managed-default` stack |
| AgentCore Gateway | ~1 min | Nothing |
| Multi-agent | ~25 min for Part 2 | Runtime and memory log groups, the `.bedrock_agentcore/` build cache |

## Troubleshooting

| You see | What is happening | Fix |
|---|---|---|
| `ExpiredToken` / `ExpiredTokenException: The security token included in the request is expired` | The tool isn't using your SSO profile (it fell back to old `[default]` credentials), or the SSO session expired | `aws sso login --profile <p>`; make sure `AWS_PROFILE` is set in that shell (`set -a; source ../../.env; set +a`) or in `.env` for notebooks (restart the kernel) |
| `ResourceNotFoundException: Model use case details have not been submitted for this account` | The Anthropic use-case form hasn't been submitted (or is still processing) for this account | Submit the form (step 2 above), or `export BEDROCK_MODEL_ID=us.amazon.nova-2-lite-v1:0` |
| `Running cells with 'Python 3.x' requires the ipykernel package` | The notebook kernel is a bare Python, not the course venv | Select Kernel → Python Environments → `.venv` (don't `pip install --user` into the system Python) |
| `FileNotFoundError: ... 'weather_openapi.json'` (or another file next to the notebook) | The kernel started in a different folder. In VS Code this happens when `jupyter.notebookFileRoot` is set to `${workspaceFolder}` | Set `"jupyter.notebookFileRoot": "${fileDirname}"` (workspace `.vscode/settings.json`), restart the kernel. Don't change the paths in the notebook |
| A script sits at `(END)` | The AWS CLI pager is showing long output | Press `q` (the script continues), or `export AWS_PAGER=""` before running |
| `ERROR: AWS SAM CLI is not installed.` | `lambda/deploy.sh` needs SAM | `brew install aws-sam-cli`, or `uv tool install aws-sam-cli` / `pipx install aws-sam-cli` |
| `curl: (6) Could not resolve host: ...elb.amazonaws.com` right after `ecs-fargate/deploy.sh` | The load balancer is still provisioning; its DNS name only works once it is active (2–4 min) | `aws elbv2 wait load-balancer-available --names hospital-scheduling-agent-alb`, then retry. A `503` after that means the task is still starting |
| ECS deploy fails looking for a VPC or subnets | The account has no default VPC | `aws ec2 create-default-vpc --region us-east-1` |
| `Failed to enable observability ... X-Ray Delivery Destination is supported with CloudWatch Logs as a Trace Segment Destination` during an AgentCore deploy | One-time warning on a fresh account while X-Ray's trace destination is still being set up | Ignore it; the deploy continues |

## Groq editions (no Bedrock models)

Every lab has a Groq edition that deploys the same agent with its model on Groq. The AWS services are unchanged, so you still need AWS credentials — but no Bedrock model access. AgentCore labs need an AWS account on the Paid plan (AgentCore isn't available on the Free plan).

| Lab | Groq edition | How the Groq key reaches the agent |
|---|---|---|
| `agentcore-runtime/` | `deploy_to_agentcore_runtime_groq.ipynb`, `agent_groq.py` | runtime environment variable |
| `ecs-fargate/` | `deploy-groq.sh`, `cleanup-groq.sh`, `app/app_groq.py`, `Dockerfile.groq` | Secrets Manager → task definition `secrets` |
| `lambda/` | `deploy-groq.sh`, `cleanup-groq.sh`, `template-groq.yaml`, `src/app_groq.py` | Secrets Manager → CloudFormation dynamic reference |
| `agentcore-gateway/` | `gateway_openapi_target_groq.ipynb` | the agent runs in the notebook |
| `multi-agent-on-agentcore/` | `stock_a2a_agent_groq.py`, `orchestrator_groq.py` (the MCP server is unchanged) | `agentcore deploy --env GROQ_API_KEY=...` |

Each lab's README has the details. Set `GROQ_API_KEY` (and optionally `GROQ_MODEL_ID`, default `openai/gpt-oss-120b`) before deploying.
