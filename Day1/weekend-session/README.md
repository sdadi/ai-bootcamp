# Agentic AI on AWS — Hands-on Labs

**by Shivank Agrawal · Future with Shivank**
refer to github at https://github.com/pulkit-scaler/weekend-sessions
Build, deploy, evaluate and govern production AI agents using the Strands Agents SDK, Amazon Bedrock and Amazon Bedrock AgentCore — from a one-line "hello agent" to multi-agent systems running in the cloud.

## Sections

| Section | Folder | What you build | Key ideas |
|---|---|---|---|
| 1 | [`section-01-your-first-agent`](section-01-your-first-agent/) | Your first agents with Strands and LangGraph | Model, tools, system prompt |
| 2 | [`section-02-agent-tools`](section-02-agent-tools/) | Tool-using agents, AWS-aware agents | `@tool`, prebuilt tools, `use_aws`, class-based and async tools |
| 3 | [`section-03-agent-memory`](section-03-agent-memory/) | Agents that remember users and conversations | Mem0 long-term memory, AgentCore Memory checkpointer |
| 4 | [`section-04-multi-agent-patterns`](section-04-multi-agent-patterns/) | Teams of agents | Supervisor–worker, agent-as-tool, graph, swarm |
| 5 | [`section-05-mcp-and-a2a`](section-05-mcp-and-a2a/) | Agents that talk to tools and to each other | MCP servers/clients, A2A protocol |
| 6 | [`section-06-production-deployment`](section-06-production-deployment/) | Agents in production | AgentCore Runtime, ECS Fargate, Lambda, AgentCore Gateway, multi-agent on AgentCore |
| 7 | [`section-07-evals-observability-governance`](section-07-evals-observability-governance/) | Trustworthy agents | AgentCore Evaluations, OpenTelemetry tracing, Bedrock Guardrails, AgentCore Policy (Cedar) |

Every section and lab folder has a `README.md`; dependencies are in a `requirements.txt` next to the code (or installed by the notebook's first cell).

---

## No Bedrock access? Use the Groq editions

New AWS accounts often can't call Bedrock models: Free-plan accounts commonly report zero Bedrock quotas, and Anthropic models need an AWS Marketplace subscription, which the Free plan restricts. So every notebook and script that uses a Bedrock model has a **`_groq` twin** next to it — the same agent code, with the model running on [Groq](https://console.groq.com) instead. Groq has an OpenAI-compatible API, so Strands' `OpenAIModel` talks to it; nothing else in the lesson changes.

```bash
pip install -r requirements-groq.txt        # in each section, instead of requirements.txt
echo "GROQ_API_KEY=gsk_..." > .env          # repo root; free key at https://console.groq.com/keys
python setup/check_setup_groq.py            # checks the key, the models, and (optionally) AWS/AgentCore
```

| Section | Groq editions | Also needs |
|---|---|---|
| 1, 4, 5 | every notebook and script (`*_groq.ipynb`, `*_groq.py`) | nothing else (Section 4 notebook 1: a Tavily key) |
| 2 | all three notebooks | notebook 2, part 3: AWS credentials (S3, DynamoDB, Lambda) |
| 3 | Mem0 notebook (embeddings run locally); AgentCore Memory notebook | notebook 2: AWS + AgentCore |
| 6 | AgentCore Runtime, ECS Fargate, Lambda, Gateway, multi-agent | AWS (+ AgentCore for Runtime, Gateway, multi-agent) |
| 7 | observability 1–5, guardrails (no AWS at all), Cedar policy, evaluation | observability 5, Cedar and evaluation: AWS + AgentCore |

- **No Bedrock anywhere.** The guardrails notebook replaces Bedrock Guardrails with a Groq safety model (`openai/gpt-oss-safeguard-20b`) plus PII rules in code; the evaluation notebook judges with a Groq model through the open-source [Strands Evals](https://github.com/strands-agents/evals) SDK instead of AgentCore Evaluations (whose judges are Bedrock models).
- **AgentCore needs a Paid-plan account.** AWS lists Amazon Bedrock AgentCore as "Paid plan exclusive" on the [Free Tier page](https://aws.amazon.com/free/). Upgrading keeps your Free Tier credits. The AgentCore labs use no Bedrock models.
- **Models.** The default is `openai/gpt-oss-120b`; change it everywhere with `export GROQ_MODEL_ID=openai/gpt-oss-20b` (or `qwen/qwen3.8-27b`).
- **Free-tier limits.** About 8,000 tokens per minute and 200,000 tokens per day *per model*. Short waits (`rate limit`) retry automatically. `Request too large ... (TPM)` means one request is bigger than 8,000 tokens — waiting won't help; start a fresh conversation or shorten the input. If you hit the daily limit, switch `GROQ_MODEL_ID` to another model or upgrade to Groq's Dev tier.
- **Run on a local model instead.** Anything that runs on your machine can use a local OpenAI-compatible server such as [Ollama](https://ollama.com): `ollama pull llama3.1`, then `export GROQ_BASE_URL=http://localhost:11434/v1 GROQ_API_KEY=ollama GROQ_MODEL_ID=llama3.1`. Small local models are slower and less reliable with tools; deployed labs (AgentCore, ECS, Lambda) can't reach a model on your laptop; and the two LangChain/LangGraph files use Groq's own client (`langchain-groq`), so they stay on Groq.

---

## 1. What you need

**Everyone**
- An AWS account (a sandbox/learning account is ideal)
- Region **us-east-1** (N. Virginia) — every AgentCore feature used here is available there
- Python **3.11 or 3.12** (3.10 minimum)
- [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html), Git, and Jupyter (JupyterLab or VS Code)

**Extra tools by section**

| Needed for | Tool / account |
|---|---|
| Section 4, notebook 1 | Free [Tavily](https://www.tavily.com/) API key (`TAVILY_API_KEY`) |
| Section 5, AWS docs MCP agent | [`uv`](https://docs.astral.sh/uv/) (`pip install uv`) |
| Section 6, ECS and Lambda | Docker (running), [AWS SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html) for Lambda |
| Section 6, multi-agent on AgentCore | `jq`, `bedrock-agentcore-starter-toolkit` |
| Section 7, observability (optional parts) | Docker (Jaeger / OTel Collector), Langfuse account, LangSmith account, Datadog account |
| Section 7, evaluations and CloudWatch tracing | CloudWatch **Transaction Search** enabled (one click, once per account/Region) |

Windows users: run the shell scripts from WSL or Git Bash.

---

## 2. Set up AWS access

### Option A — Your laptop (VS Code, Kiro, Cursor, JupyterLab)

```bash
aws configure                 # access key, secret, region = us-east-1
# or, with IAM Identity Center:
aws configure sso && aws sso login --profile my-profile && export AWS_PROFILE=my-profile
```

### Option B — Amazon SageMaker Studio (JupyterLab space)

Credentials come from the Studio execution role. Use an `ml.m5.large` (or bigger) space with at least 20 GB of storage. Attach the permissions below to the execution role.

### Permissions

Use a **sandbox / learning account**, not a production one. There, an administrator user is simplest. Otherwise attach [`setup/iam-policy-course.json`](setup/iam-policy-course.json) to your user or role — it covers every lab in the course. It is deliberately broad (the labs create IAM roles, gateways, Lambda functions and more), so treat it as sandbox-only.

### Model access

Amazon Bedrock serverless models are available by default — no request form. The labs use:

| Model | ID used in code |
|---|---|
| Claude Sonnet 4.6 (default) | `global.anthropic.claude-sonnet-4-6` |
| Claude Haiku 4.5 (fast/cheap: memory, evaluation judge, Lambda) | `global.anthropic.claude-haiku-4-5-20251001-v1:0` |
| Titan Text Embeddings v2 (Section 3 memory) | `amazon.titan-embed-text-v2:0` |

The `global.` prefix is a cross-Region inference profile: Bedrock routes each request to the best Region automatically, and it works from any supported source Region (including India).

**Change the model everywhere** without editing code:

```bash
export BEDROCK_MODEL_ID=global.anthropic.claude-haiku-4-5-20251001-v1:0
```

---

## 3. Install and check

```bash
cd future-with-shivank-agentic-ai-on-aws
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install --upgrade pip boto3

export AWS_REGION=us-east-1
export AWS_DEFAULT_REGION=us-east-1

python setup/check_setup.py
```

You should see a ✅ for both models. Then, for each section:

```bash
cd section-01-your-first-agent
pip install -r requirements.txt
jupyter lab            # or open the folder in VS Code
```

One virtual environment for the whole course works fine. If you ever hit a dependency clash, create a fresh venv for that section.

---

## 4. Costs and cleanup

Most labs cost cents. The ones that create cloud resources end with a **Clean up** cell or a `cleanup.sh` — run it before you move on:

| Lab | Creates | Clean up with |
|---|---|---|
| 2 · `02_prebuilt_and_aws_tools` | S3 bucket, DynamoDB table, Lambda | last notebook cell |
| 3 · AgentCore Memory | Memory resource | last notebook cell |
| 6 · AgentCore Runtime / Gateway | Runtime, ECR repo, gateway, roles | last notebook cell |
| 6 · ECS Fargate | ALB, ECS service, ECR, roles | `./cleanup.sh` |
| 6 · Lambda | CloudFormation stack | `./cleanup.sh` |
| 6 · Multi-agent | 3 runtimes, Cognito pool | commands at the end of its README |
| 7 · Evaluation, Guardrails, Policy, CloudWatch | Runtime, evaluator, guardrail, gateway, policy engine, log group | last notebook cell |

---

## 5. Troubleshooting

| Error | Fix |
|---|---|
| `AccessDeniedException` calling Bedrock | Check IAM permissions and that `AWS_REGION` is set |
| `ValidationException ... model identifier is invalid` / "on-demand throughput isn't supported" | Use an inference-profile ID (with `global.` or `us.`), as in the table above |
| `ResourceNotFoundException ... Legacy model` / end-of-life | That model was retired — set `BEDROCK_MODEL_ID` to a current model |
| `ThrottlingException` | Wait a minute and retry; use Haiku for bulk runs |
| `ModuleNotFoundError` in a notebook | Run the `%pip install -r requirements.txt` cell, then restart the kernel |
| Port already in use (Sections 5–7) | Stop the old process (`lsof -i :9001`) or `docker rm -f <name>` |

---

© 2026 Shivank Agrawal · Future with Shivank. Code is provided under the MIT License — see [`LICENSE`](LICENSE).
