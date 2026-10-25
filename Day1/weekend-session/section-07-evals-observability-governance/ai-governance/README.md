# AI Governance

| Item | What it covers |
|---|---|
| `01_bedrock_guardrails.ipynb` | Create a guardrail (content filters, denied topic, PII), attach it to an agent, test it — then show that guardrails can't stop a $50,000 refund tool call |
| `02_agentcore_policy_cedar.ipynb` | Put the refund tool behind an AgentCore Gateway, attach a Cedar policy engine, and watch $500 succeed while $50,000 is denied |
| `cedar-policies/` | Four Cedar rules: allow the refund tool, forbid refunds > $1,000, physician-only patient records, organisation data boundary |
| `governance_checklist.md` | The human layer: agent card, risk assessment, audit trail, incident response |

## Run

```bash
pip install -r requirements.txt
jupyter lab
```

IAM: `bedrock:CreateGuardrail`/`GetGuardrail`/`DeleteGuardrail`, `bedrock:InvokeModel`; for notebook 2 also `bedrock-agentcore:*` (gateway, policy), `lambda:*` on the demo function, and IAM role management.

## Cost

Minimal. Guardrails are billed per 1,000 text units. Both notebooks delete what they create in their last cell.

## Run on Groq instead of Bedrock

```bash
pip install -r requirements-groq.txt
```

- `01_guardrails_groq.ipynb` needs **only a Groq key** — no AWS. It builds the same guardrail (content filters, a denied topic, PII) without Bedrock: Groq's safety model `openai/gpt-oss-safeguard-20b` checks each message against a plain-language policy, PII is matched in code, and Strands hooks apply it to the agent's input and output. On the free tier the safety model allows 2,000 tokens/minute (about two checks a minute), so the notebook takes a few minutes; set `GUARDRAIL_MODEL_ID=openai/gpt-oss-20b` for faster checks.
- `02_agentcore_policy_cedar_groq.ipynb` keeps AgentCore Gateway and Policy (AWS credentials, no Bedrock model access); only the agent's model changes.
