# Cedar policy examples for AgentCore Policy

AgentCore Policy evaluates every tool call that passes through an AgentCore
Gateway against Cedar rules. The rules run outside the model, so no prompt can
talk its way past them.

How AgentCore names things in Cedar:

| Concept | Cedar syntax |
|---|---|
| The tool being called | `action == AgentCore::Action::"<TargetName>___<tool_name>"` |
| The gateway it goes through | `resource == AgentCore::Gateway::"<gateway ARN>"` |
| The tool's input arguments | `context.input.<argument>` |
| Caller with an OAuth token (`CUSTOM_JWT` gateway) | `principal is AgentCore::OAuthUser`, claims via `principal.hasTag("x")` / `principal.getTag("x")` |
| Caller with IAM credentials (`AWS_IAM` gateway) | `principal is AgentCore::IamEntity`, ARN via `principal.id` |

Evaluation rules:
- **Default deny** — once a policy engine is attached in `ENFORCE` mode, a tool call is allowed only if some `permit` matches.
- **Forbid wins** — if any `forbid` matches, the call is denied even if a `permit` also matches.

| File | Rule |
|---|---|
| `01_allow_refund_tool.cedar` | Permit the refund tool (needed because of default deny) |
| `02_forbid_large_refunds.cedar` | Deny refunds over $1,000 |
| `03_patient_records_physician_only.cedar` | Only a physician from the same department can read a patient record |
| `04_org_data_boundary.cedar` | Users may only query their own organisation's data |

Replace `<GATEWAY_ARN>` before using a file. `02_agentcore_policy_cedar.ipynb` shows files 01 and 02 working end to end; 03 and 04 need a gateway with OAuth (`CUSTOM_JWT`) so the caller's claims are available.
