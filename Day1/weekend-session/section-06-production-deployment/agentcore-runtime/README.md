# AgentCore Runtime — Deploy a Strands Agent

| File | Purpose |
|---|---|
| `agent.py` | The agent: a normal Strands agent + 4 lines of `BedrockAgentCoreApp` boilerplate |
| `requirements.txt` | Packages installed **inside** the deployed container |
| `deploy_to_agentcore_runtime.ipynb` | Configure → launch → invoke (toolkit and boto3) → stop session → destroy |
| `Dockerfile` | Reference copy of what the starter toolkit generates (it is regenerated on configure) |

## Test locally

```bash
pip install bedrock-agentcore strands-agents
python agent.py
# in another terminal
curl -X POST http://localhost:8080/invocations -H "Content-Type: application/json" \
  -d '{"prompt": "What are the key clauses in a standard NDA?"}'
```

## Deploy

Open `deploy_to_agentcore_runtime.ipynb` and run the cells in order. The build runs in AWS CodeBuild (ARM64), so you don't need Docker.

Prefer the CLI? The same flow:

```bash
pip install bedrock-agentcore-starter-toolkit
agentcore configure -e agent.py --non-interactive
agentcore deploy
agentcore invoke '{"prompt": "Summarize the termination clause risks"}'
agentcore destroy
```

## Groq edition (no Bedrock models)

| File | Purpose |
|---|---|
| `agent_groq.py` | The same agent with its model on Groq (`GROQ_API_KEY`, `GROQ_MODEL_ID`) |
| `requirements-groq.txt` | Packages installed inside the deployed container |
| `deploy_to_agentcore_runtime_groq.ipynb` | The same configure → launch → invoke → destroy lifecycle |

```bash
pip install bedrock-agentcore strands-agents[openai] python-dotenv
python agent_groq.py          # local test on :8080, same curl as above
```

The notebook builds from a separate `build_groq/` folder: the starter toolkit reuses any `Dockerfile` it finds, and the one in this folder is for `agent.py`. The launch passes `GROQ_API_KEY` to the runtime as an environment variable — fine for a lab; in production keep it in Secrets Manager or an AgentCore Identity API-key provider. You still need AWS credentials for AgentCore, ECR and CodeBuild, but no Bedrock model access. Takes about 3 minutes end to end.
