# AgentCore Runtime — Deploy a Strands Agent

| File | Purpose |
|---|---|
| `agent.py` | The agent: a normal Strands agent + 4 lines of `BedrockAgentCoreApp` boilerplate |
| `requirements.txt` | Packages installed **inside** the deployed container |
| `deploy_to_agentcore_runtime.ipynb` | Configure → launch → invoke (toolkit and boto3) → stop session → destroy |
| `Dockerfile` | Reference copy of what the starter toolkit generates (it is regenerated on configure) |

## Test locally

`agent.py` calls Bedrock from your machine, so the terminal needs your AWS profile and a model you can use (see "Before you start" in the section README):

```bash
pip install bedrock-agentcore strands-agents
set -a; source ../../.env; set +a      # AWS_PROFILE, AWS_REGION, and BEDROCK_MODEL_ID if you need a non-Anthropic model
python agent.py
# in another terminal
curl -X POST http://localhost:8080/invocations -H "Content-Type: application/json" \
  -d '{"prompt": "What are the key clauses in a standard NDA?"}'
```

The server logs the model it uses on every call (`Model id: ...`). If the call fails, the error is in that log:

- `ExpiredTokenException` → the terminal isn't using your SSO profile (or the session expired): `aws sso login`, then the `source` line again, then restart `python agent.py`.
- `Model use case details have not been submitted` → Anthropic access isn't enabled in this account yet; set `BEDROCK_MODEL_ID=us.amazon.nova-2-lite-v1:0` and restart.

Stop the server with `Ctrl+C` when done (it holds port 8080).

## Deploy

Open `deploy_to_agentcore_runtime.ipynb` and run the cells in order. The build runs in AWS CodeBuild (ARM64), so you don't need Docker. The whole notebook takes about 3 minutes.

| Step | What happens | What you should see |
|---|---|---|
| Configure | The toolkit records the entrypoint, region and an execution role (created on launch) | `Configuring BedrockAgentCore agent: doc_analysis_agent`, `Using existing Dockerfile` |
| Launch | Your folder is zipped to S3, CodeBuild builds an ARM64 image, pushes it to ECR, and the runtime is created. `BEDROCK_MODEL_ID` from your environment is passed to the runtime | `CodeBuild completed successfully`, then the agent ID and ARN |
| Wait for READY | Polls the endpoint status | `Status: READY` |
| Invoke (toolkit) | Calls the runtime with a contract clause | An analysis of the clause |
| Invoke (boto3) | The same call with plain `boto3`, in a new session | A session ID and a response |
| Stop session | Ends the microVM for that session | `Session ... stopped.` |
| Destroy | Deletes the runtime, ECR repository, CodeBuild project, IAM roles and S3 artifacts | `Removed ... resources`, `Errors: 0` |

What to expect along the way:

- **A warning on fresh accounts:** launch may print `Failed to enable observability ... X-Ray Delivery Destination is supported with CloudWatch Logs as a Trace Segment Destination`. X-Ray is still being set up in the account; the deploy carries on and works.
- **The toolkit invoke prints only the first chunk of the answer.** The cell prints `response["response"][0]`, and the response body arrives in 1 KB chunks, so a long answer shows up cut off, wrapped in quotes and with literal `\n`. To see the whole answer: `print("".join(response["response"]))`.
- **The boto3 invoke asks about "this contract" without sending one.** It's a fresh session, so the agent asks you for the contract text (or, with some models, makes one up). Put a contract excerpt in the prompt to get a real summary.
- **What destroy keeps:** CloudWatch log groups (`/aws/bedrock-agentcore/runtimes/doc_analysis_agent-<id>-DEFAULT` and `/aws/codebuild/bedrock-agentcore-doc_analysis_agent-builder`) and the runtime's trace delivery settings. They cost next to nothing; to remove the log groups:
  ```bash
  aws logs describe-log-groups --log-group-name-prefix /aws/bedrock-agentcore/runtimes/doc_analysis_agent --query 'logGroups[].logGroupName' --output text
  aws logs delete-log-group --log-group-name <name>
  ```

Prefer the CLI? The same flow:

```bash
pip install bedrock-agentcore-starter-toolkit
agentcore configure -e agent.py --non-interactive
agentcore deploy --env BEDROCK_MODEL_ID=${BEDROCK_MODEL_ID:-global.anthropic.claude-sonnet-4-6}
agentcore invoke '{"prompt": "Summarize the termination clause risks"}'
agentcore destroy --delete-ecr-repo      # without the flag, the ECR repository is kept
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
