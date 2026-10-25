# Section 6 — Shipping Agents to Production

| Folder | Target | Example |
|---|---|---|
| `agentcore-runtime/` | Amazon Bedrock AgentCore Runtime | Contract-analysis agent deployed with the starter toolkit (no Docker needed) |
| `ecs-fargate/` | Amazon ECS on Fargate behind an ALB | Hospital surgical-scheduling agent with a chat UI |
| `lambda/` | AWS Lambda behind API Gateway | Serverless contract-analysis agent |
| `agentcore-gateway/` | AgentCore Gateway | Turn a REST API (OpenAPI spec in S3) into MCP tools for an agent |
| `multi-agent-on-agentcore/` | AgentCore Runtime (MCP + A2A) | Finance orchestrator → stock agent (A2A) → stock tools (MCP) |
| `setup_cognito.sh` | Cognito | OAuth user pool + token for the multi-agent deployment |

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
- `ecs-fargate/`: Docker running
- `lambda/`: Docker running and the AWS SAM CLI

## Quick start

```bash
# AgentCore Runtime
cd agentcore-runtime && jupyter lab deploy_to_agentcore_runtime.ipynb

# ECS Fargate
cd ecs-fargate && chmod +x deploy.sh cleanup.sh && ./deploy.sh      # ./cleanup.sh when done

# Lambda
cd lambda && chmod +x deploy.sh cleanup.sh && ./deploy.sh           # ./cleanup.sh when done

# AgentCore Gateway
cd agentcore-gateway && jupyter lab gateway_openapi_target.ipynb

# Multi-agent (local first, then AgentCore) — see multi-agent-on-agentcore/README.md
```

All three agents default to `global.anthropic.claude-sonnet-4-6`; set `BEDROCK_MODEL_ID` to change it (the deploy scripts pass it through).

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
