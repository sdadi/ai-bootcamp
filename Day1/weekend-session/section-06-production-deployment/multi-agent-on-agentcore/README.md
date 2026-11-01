# Multi-Agent Finance System: MCP + A2A on Amazon Bedrock AgentCore

One MCP server for stock-price tools, one A2A sub-agent, one A2A orchestrator.

```
    ┌──────────────────┐
    │    Finance       │
    │  Orchestrator    │   A2A, port 9000
    └────────┬─────────┘
             │ A2A  (tool: ask_stock_agent)
             ▼
    ┌──────────────────┐
    │   Stock Agent    │   A2A, port 9001 locally (9000 on AgentCore)
    │  + http_request  │
    └────────┬─────────┘
             │ MCP (streamable HTTP)
             ▼
    ┌──────────────────┐
    │   Stock Tools    │   MCP, port 8000, path /mcp
    │ yfinance prices  │
    └──────────────────┘
```

| File | Protocol | Local port | What it does |
|---|---|---|---|
| `stock_mcp_server.py` | MCP | 8000 | `get_stock_price` tool backed by yfinance |
| `stock_a2a_agent.py` | A2A | 9001 | Agent with the MCP tools + `http_request` |
| `orchestrator.py` | A2A | 9000 | Agent that calls the Stock Agent through an `ask_stock_agent` tool |

## Part 1 — Run everything locally

The two agents call Bedrock from your machine, so Terminals 2 and 3 need your AWS profile and a model you can use (see "Before you start" in the section README). Start them in this order, each in its own terminal, from this folder:

```bash
pip install -r requirements.txt

# Terminal 1 — wait for "Uvicorn running on http://0.0.0.0:8000" before going on
python stock_mcp_server.py

# Terminal 2
set -a; source ../../.env; set +a      # AWS_PROFILE, AWS_REGION, and BEDROCK_MODEL_ID if needed
python stock_a2a_agent.py

# Terminal 3
set -a; source ../../.env; set +a
python orchestrator.py
```

- **Wait for the MCP server.** The stock agent connects to it as it starts. The server's first start can take ~30 s; if Terminal 2 starts too early it exits with a connection error (`ConnectError: All connection attempts failed` / `MCPClientInitializationError`) — just start it again.
- **Another app on port 8000?** The MCP server always uses 8000 (AgentCore expects it there). On macOS it can still start next to an app listening on `127.0.0.1:8000`, but `localhost:8000` then reaches the other app, and the stock agent fails with `POST http://localhost:8000/mcp "HTTP/1.1 404 Not Found"` followed by `McpError: Session terminated`. See what's listening with `lsof -nP -iTCP:8000 -sTCP:LISTEN`, and stop the other app. Or, for the local test only, run the MCP server on another port and point the stock agent at it:
  ```bash
  # Terminal 1
  python -c "import stock_mcp_server as s; s.mcp.settings.port = 8010; s.mcp.run(transport='streamable-http')"
  # Terminal 2, before python stock_a2a_agent.py
  export STOCK_MCP_URL=http://localhost:8010/mcp
  ```
- **`Found credentials in shared credentials file` in Terminal 2 or 3** means `AWS_PROFILE` isn't set in that terminal, so calls will use your old default credentials — run the `source` line first.

Talk to the orchestrator (Terminal 4):

```bash
curl -s -X POST http://localhost:9000/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0", "id": "1", "method": "message/send",
    "params": {"message": {"role": "user",
      "parts": [{"kind": "text", "text": "How is NVDA stock doing today?"}],
      "messageId": "test-1"}}
  }' | jq -r 'if .result.artifacts then [.result.artifacts[0].parts[].text] | join("") else (.error // .result.status) end'
```

Or talk to the Stock Agent directly by changing the port to `9001`. Expect something like "NVIDIA Corporation (NVDA) is currently trading at **$…**, down **…%**". The agents stream their answer, so it arrives split into many `parts`; the `join("")` above glues them back together (`.parts[0].text` alone prints only the first word or letter). If you get `{"state":"failed", ... "Agent execution failed"}` instead, the agent hit an error and the reason is printed in Terminal 2 or 3. Most often that terminal is missing your AWS profile or model (`ExpiredToken`, or the model-access message): stop it, run the `source` line, and start it again.

## Part 2 — Deploy all three to AgentCore Runtime

You need the starter toolkit CLI (`pip install bedrock-agentcore-starter-toolkit` — note: AWS also ships a separate Node.js `agentcore` CLI with different commands; this lab uses the Python one) and `jq`.

**Paste each block below as it is — there's nothing to edit.** Run them from this lab folder, with your AWS login working (see "Before you start" in the section README). Every block starts with `source session.sh`, which:

- loads the Cognito pool from Step 1 (saved in `.env.cognito` in this folder, git-ignored) and fetches a fresh token, so a block works in any terminal — the same one, a new one, or tomorrow;
- sets `$AUTH_CONFIG`, the Cognito authorizer passed to `agentcore configure`;
- provides `runtime_url <name>`, which looks up a deployed agent's URL by its name.

The commands in a block are chained with `&&`, so if something is missing — an earlier step not done, an expired AWS login — the block stops with a message instead of deploying a half-configured agent.

It takes about 25 minutes: each deploy is 2–6 minutes (`configure` also sets up an AgentCore memory for each agent, which `destroy` removes). Every `agentcore` command prints a "Starter Toolkit CLI is no longer supported" recommendation; this lab still works with it — `export AGENTCORE_SUPPRESS_RECOMMENDATION=1` hides the message. Deploys may also print a one-time `Failed to enable observability ... X-Ray Delivery Destination` warning on a fresh account; the deploy continues. If Claude isn't enabled in your account, `export BEDROCK_MODEL_ID=us.amazon.nova-2-lite-v1:0` first (or set it in `.env`) — the deploy commands pass it to the agents.

### Step 1: Cognito user pool for OAuth

Pick your own password (8+ characters):

```bash
export REGION=us-east-1 COGNITO_USERNAME=testuser COGNITO_PASSWORD='YourPassword123!'
source session.sh
```

It prints the pool ID, client ID, discovery URL and the start of a token, and saves the pool's details in `.env.cognito`. Running it again reuses the same pool — it never creates a second one. The pool's access tokens last a day (Cognito's default is an hour) — Step 3 explains why.

### Step 2: Deploy the MCP server

```bash
source session.sh && \
agentcore configure -e stock_mcp_server.py --protocol MCP \
  --requirements-file requirements.txt --authorizer-config "$AUTH_CONFIG" \
  --region "$REGION" --non-interactive && \
agentcore deploy --agent stock_mcp_server
```

In the configure summary, **Authorization must not say `IAM (default)`** — that would mean the Cognito authorizer wasn't passed (see Troubleshooting below).

### Step 3: Deploy the Stock Agent

On AgentCore, A2A agents listen on port 9000, so the deploy passes `PORT=9000`. It also passes the MCP server's URL and your token (`AGENT_BEARER_TOKEN`), which the agent uses to call the MCP server. That token is fixed into the agent when you deploy, and the orchestrator gets one the same way in Step 4, so the deployed chain works for a day after these deploys; after that, give the agents fresh tokens with the block under Step 5.

```bash
source session.sh && \
STOCK_MCP_URL=$(runtime_url stock_mcp_server mcp) && \
agentcore configure -e stock_a2a_agent.py --protocol A2A \
  --requirements-file requirements.txt --authorizer-config "$AUTH_CONFIG" \
  --region "$REGION" --non-interactive && \
agentcore deploy --agent stock_a2a_agent \
  --env PORT=9000 --env "STOCK_MCP_URL=$STOCK_MCP_URL" --env "AGENT_BEARER_TOKEN=$BEARER_TOKEN" \
  --env "BEDROCK_MODEL_ID=${BEDROCK_MODEL_ID:-global.anthropic.claude-sonnet-4-6}"
```

Deploy once more so the agent card advertises the agent's public URL (A2A clients send their messages to the URL in the card, and that URL only exists after the first deploy):

```bash
source session.sh && \
STOCK_MCP_URL=$(runtime_url stock_mcp_server mcp) && \
STOCK_A2A_URL=$(runtime_url stock_a2a_agent) && \
agentcore deploy --agent stock_a2a_agent --auto-update-on-conflict \
  --env PORT=9000 --env "STOCK_MCP_URL=$STOCK_MCP_URL" --env "AGENT_BEARER_TOKEN=$BEARER_TOKEN" \
  --env "BEDROCK_MODEL_ID=${BEDROCK_MODEL_ID:-global.anthropic.claude-sonnet-4-6}" \
  --env "AGENTCORE_RUNTIME_URL=$STOCK_A2A_URL"
```

### Step 4: Deploy the Orchestrator

```bash
source session.sh && \
STOCK_A2A_URL=$(runtime_url stock_a2a_agent) && \
agentcore configure -e orchestrator.py --protocol A2A \
  --requirements-file requirements.txt --authorizer-config "$AUTH_CONFIG" \
  --region "$REGION" --non-interactive && \
agentcore deploy --agent orchestrator \
  --env "STOCK_A2A_URL=$STOCK_A2A_URL" --env "AGENT_BEARER_TOKEN=$BEARER_TOKEN" \
  --env "BEDROCK_MODEL_ID=${BEDROCK_MODEL_ID:-global.anthropic.claude-sonnet-4-6}"
```

### Step 5: Test

```bash
source session.sh && \
ORCH_URL=$(runtime_url orchestrator) && \
curl -s -X POST "${ORCH_URL}?qualifier=DEFAULT" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $BEARER_TOKEN" \
  -H "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id: $(uuidgen)$(uuidgen)" \
  -d '{
    "jsonrpc": "2.0", "id": "1", "method": "message/send",
    "params": {"message": {"role": "user",
      "parts": [{"kind": "text", "text": "How is NVDA stock doing today?"}],
      "messageId": "test-1"}}
  }' | jq -r 'if .result.artifacts then [.result.artifacts[0].parts[].text] | join("") else (.error // .result.status) end'
```

Expect something like "NVIDIA Corporation (NVDA) is currently trading at **$…**, down **…%**". The first call takes 30–45 s (three cold starts in a row). The agents stream their answer in many `parts`; the `jq` filter joins them, and prints the failure status instead if there's no answer (replace it with `jq .` to see the raw response).

To confirm the whole chain ran, tail each runtime's log group, `aws logs tail /aws/bedrock-agentcore/runtimes/<agent-id>-DEFAULT --since 10m` (the agent ID is the last part of its ARN): the orchestrator logs a POST to the stock agent's URL, the stock agent logs `POST /`, and the MCP server logs a `CallToolRequest`. A `401` in those logs means the token fixed into an agent at deploy time has expired (a day after that deploy).

**Refresh the agents' tokens** (a day after Steps 3–4, or whenever Step 5 reports a `401`). This redeploys both agents in place with a fresh token. It deliberately skips `agentcore configure`: re-running configure on an agent that's already deployed makes `agentcore destroy` keep that agent's memory later.

```bash
source session.sh && \
STOCK_MCP_URL=$(runtime_url stock_mcp_server mcp) && \
STOCK_A2A_URL=$(runtime_url stock_a2a_agent) && \
agentcore deploy --agent stock_a2a_agent --auto-update-on-conflict \
  --env PORT=9000 --env "STOCK_MCP_URL=$STOCK_MCP_URL" --env "AGENT_BEARER_TOKEN=$BEARER_TOKEN" \
  --env "BEDROCK_MODEL_ID=${BEDROCK_MODEL_ID:-global.anthropic.claude-sonnet-4-6}" \
  --env "AGENTCORE_RUNTIME_URL=$STOCK_A2A_URL" && \
agentcore deploy --agent orchestrator --auto-update-on-conflict \
  --env "STOCK_A2A_URL=$STOCK_A2A_URL" --env "AGENT_BEARER_TOKEN=$BEARER_TOKEN" \
  --env "BEDROCK_MODEL_ID=${BEDROCK_MODEL_ID:-global.anthropic.claude-sonnet-4-6}"
```

### Step 6: Clean up

```bash
agentcore destroy --agent orchestrator --force --delete-ecr-repo
agentcore destroy --agent stock_a2a_agent --force --delete-ecr-repo
agentcore destroy --agent stock_mcp_server --force --delete-ecr-repo
# Delete any of this lab's memories that destroy kept (see below)
for id in $(aws bedrock-agentcore-control list-memories --region "${REGION:-us-east-1}" \
    --query "memories[?status=='ACTIVE' && (starts_with(id,'orchestrator_mem-') || starts_with(id,'stock_a2a_agent_mem-') || starts_with(id,'stock_mcp_server_mem-'))].id" \
    --output text); do
  aws bedrock-agentcore-control delete-memory --memory-id "$id" --region "${REGION:-us-east-1}" > /dev/null && echo "Deleted memory $id"
done
source session.sh && aws cognito-idp delete-user-pool --user-pool-id "$POOL_ID" --region "$REGION" && rm .env.cognito
```

`agentcore destroy` removes each runtime with its IAM role, memory, workload identity and uploaded code. One exception: if `agentcore configure` was run again on an agent after its first deploy, `destroy` prints `Memory ... preserved (was pre-existing)` and keeps that memory. The loop after the destroys deletes those; when there are none, it prints nothing. Kept, so you can read them later — delete them in the CloudWatch console (or with `aws logs delete-log-group`) when you no longer need them:

- the runtimes' log groups, `/aws/bedrock-agentcore/runtimes/<agent-id>-DEFAULT`
- the memories' log groups, `/aws/vendedlogs/bedrock-agentcore/memory/APPLICATION_LOGS/<memory-id>`
- log delivery settings named after the runtime and memory IDs (free)

Leave the account-level setup the toolkit made on your first deploy — the `bedrock-agentcore-codebuild-sources-<account>-<region>` S3 bucket, the `aws/spans` log group and the `TransactionSearchXRayAccess` logs policy — it's reused by every AgentCore lab. The toolkit also leaves a build cache, `.bedrock_agentcore/` (~300 MB), in this folder; it's git-ignored and safe to delete.

### Troubleshooting Part 2

| You see | What is happening | Fix |
|---|---|---|
| `Error: set REGION, COGNITO_USERNAME and COGNITO_PASSWORD first.` | Step 1 hasn't run in this folder yet (there's no `.env.cognito`) | Run Step 1 |
| `Error: no AgentCore runtime named '...' — deploy it first.` | The step before hasn't been run, or its deploy failed | Run that step; `agentcore status --agent <name>` shows its state |
| `Error: can't reach the user pool saved in .env.cognito` | Your AWS login expired, or the pool was deleted | `aws sso login --profile <p>`; if the pool is gone, `rm .env.cognito` and run Step 1 again |
| `Error: run this from the multi-agent-on-agentcore folder.` | The terminal is in another folder | `cd` into this folder |
| The configure summary says `Authorization: IAM (default)` | `agentcore configure` ran without the Cognito authorizer (outside these blocks, or with an empty `$AUTH_CONFIG`) — callers with a bearer token will be rejected | `agentcore destroy --agent <name> --force --delete-ecr-repo`, then run its step again |
| `ExpiredTokenException` from an `aws` or `agentcore` command | The terminal isn't using your SSO profile, or the session expired | See the section README's troubleshooting table |
| Step 5 prints `"code": -32001, "message": "Token has expired"` | The token in your request has expired, so AgentCore rejects it before the orchestrator runs (nothing shows in its logs). The Step 5 block fetches a fresh one each time; this happens when a request reuses `$BEARER_TOKEN` set by an earlier command | Paste Step 5 as written |
| Step 5 answers with an apology instead of a price ("…returned a 401 Unauthorized error…"); the orchestrator's log shows `GET …stock_a2a_agent…/.well-known/agent-card.json "HTTP/1.1 401 Unauthorized"` | The token fixed into the orchestrator or the Stock Agent at deploy time has expired (a day after that deploy) | Paste the refresh block under Step 5, then Step 5 |
| `agentcore destroy` warns `Memory ... preserved (was pre-existing)`, and the memory is still listed in the console afterwards | `agentcore configure` was run again on that agent after its first deploy, so the toolkit treats the memory as one it didn't create, and `destroy` leaves it | The memory loop in Step 6 deletes it |

## Notes

- **Ports on AgentCore:** HTTP agents use 8080, MCP servers 8000 (`/mcp`), A2A agents 9000. `stock_a2a_agent.py` switches to 9000 automatically inside the AgentCore container.
- **Agent-to-agent auth:** for a classroom demo the deployed agents reuse your Cognito access token (`AGENT_BEARER_TOKEN`), fixed in at deploy time. `setup_cognito.sh` makes those tokens last a day (Cognito's maximum) so the chain outlasts a lab session. In production, give each agent its own machine identity with AgentCore Identity (OAuth2 client-credentials) instead.
- **Model:** set `BEDROCK_MODEL_ID` (e.g. with `--env BEDROCK_MODEL_ID=...`) to change the model.

## Groq edition (no Bedrock models)

`stock_a2a_agent_groq.py` and `orchestrator_groq.py` are the two agents with their model on Groq; `stock_mcp_server.py` has no model and is unchanged. Install `requirements-groq.txt` and set `GROQ_API_KEY` (and optionally `GROQ_MODEL_ID`).

**Local:** the same three terminals, with `python stock_a2a_agent_groq.py` and `python orchestrator_groq.py`. Start the MCP server first and wait for "Uvicorn running" — otherwise the stock agent exits with `MCPClientInitializationError`.

**AgentCore:** the same steps, with the Groq entrypoints and two extra `--env` flags on each agent deploy. Agent names come from the file names, so every `--agent` flag uses `stock_a2a_agent_groq` / `orchestrator_groq`:

```bash
export GROQ_MODEL_ID=openai/gpt-oss-120b     # GROQ_API_KEY comes from .env (or: export GROQ_API_KEY=...)

# Steps 1–2 — unchanged (optionally add --disable-memory --deployment-type container to the MCP server's configure too)

# Step 3 — Stock Agent (the block stops if GROQ_API_KEY is empty)
source session.sh && [ -n "$GROQ_API_KEY" ] && \
STOCK_MCP_URL=$(runtime_url stock_mcp_server mcp) && \
agentcore configure -e stock_a2a_agent_groq.py --protocol A2A \
  --requirements-file requirements-groq.txt --authorizer-config "$AUTH_CONFIG" \
  --region "$REGION" --non-interactive --disable-memory --deployment-type container && \
agentcore deploy --agent stock_a2a_agent_groq \
  --env PORT=9000 --env "STOCK_MCP_URL=$STOCK_MCP_URL" --env "AGENT_BEARER_TOKEN=$BEARER_TOKEN" \
  --env "GROQ_API_KEY=$GROQ_API_KEY" --env "GROQ_MODEL_ID=$GROQ_MODEL_ID"
# ... then Step 3's second block with `runtime_url stock_a2a_agent_groq`, `--agent stock_a2a_agent_groq` and the same two GROQ flags

# Step 4 — Orchestrator
source session.sh && [ -n "$GROQ_API_KEY" ] && \
STOCK_A2A_URL=$(runtime_url stock_a2a_agent_groq) && \
agentcore configure -e orchestrator_groq.py --protocol A2A \
  --requirements-file requirements-groq.txt --authorizer-config "$AUTH_CONFIG" \
  --region "$REGION" --non-interactive --disable-memory --deployment-type container && \
agentcore deploy --agent orchestrator_groq \
  --env "STOCK_A2A_URL=$STOCK_A2A_URL" --env "AGENT_BEARER_TOKEN=$BEARER_TOKEN" \
  --env "GROQ_API_KEY=$GROQ_API_KEY" --env "GROQ_MODEL_ID=$GROQ_MODEL_ID"

# Step 5 — as above, with `runtime_url orchestrator_groq`

# Step 6 — clean up
agentcore destroy --agent orchestrator_groq --force --delete-ecr-repo
agentcore destroy --agent stock_a2a_agent_groq --force --delete-ecr-repo
agentcore destroy --agent stock_mcp_server --force --delete-ecr-repo
# + Step 6's memory loop, if destroy said "Memory ... preserved" (the Groq agents have no memory; the MCP server does)
source session.sh && aws cognito-idp delete-user-pool --user-pool-id "$POOL_ID" --region "$REGION" && rm .env.cognito
```

- The answer streams back in many parts; to print all of it use `jq -r 'if .result.artifacts then [.result.artifacts[0].parts[].text] | join("") else (.error // .result.status) end'` instead of `.parts[0].text` (it prints the failure status if there's no answer).
- `GROQ_API_KEY` is stored as a runtime environment variable (visible in the console); for production use Secrets Manager or AgentCore Identity.
- `--deployment-type container` builds each image in CodeBuild, so only your source (a few KB) is uploaded; the default direct code deploy uploads a ~100 MB zip per agent, which is painful on a slow connection. `--disable-memory` skips the short-term memory the toolkit would otherwise create for each agent (~2.5 minutes; this lab doesn't use it).
- The Cognito token is fixed into each agent at deploy time and lasts a day; `source session.sh` fetches a fresh one at the start of every block. After a day, re-run Steps 3 and 4 (Groq versions) to refresh the agents' tokens.
- Step 6 leaves the runtime and CodeBuild log groups and the trace delivery sources/destinations; delete them in the CloudWatch console if you don't need them. Keep the account-level setup (the `bedrock-agentcore-codebuild-sources-<account>-<region>` S3 bucket, the `aws/spans` log group and the `TransactionSearchXRayAccess` logs policy) — every AgentCore lab reuses it.
- Tested end to end on AgentCore Runtime with `openai/gpt-oss-120b`: orchestrator → Stock Agent → MCP server, one request traced through all three runtimes.
