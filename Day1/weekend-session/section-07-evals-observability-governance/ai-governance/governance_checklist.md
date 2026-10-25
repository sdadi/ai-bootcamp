# Agent Governance Checklist

Technology enforces the rules. People stay accountable. Use this before any agent goes to production.

## 1. Decide whether to build it
- [ ] What decisions will the agent make, and what happens when it gets one wrong?
- [ ] What data does it touch? Which regulations apply (DPDP Act, GDPR, HIPAA, PCI DSS, sector rules)?
- [ ] Who owns the agent, and who is accountable when it fails?
- [ ] What is the human fallback?

## 2. Agent card (keep it next to the code)

```markdown
# Agent card: <agent name>
Owner:            <team / person>
Purpose:          <what it does, for whom>
Out of scope:     <what it must refuse>
Model(s):         <model IDs and versions>
Tools:            <each tool, what it can change, who can call it>
Data accessed:    <sources, sensitivity>
Guardrails:       <guardrail ID/version, policies enabled>
Cedar policies:   <policy engine, rules, ENFORCE or LOG_ONLY>
Evaluation:       <evaluators, datasets, last scores, pass thresholds>
Known limits:     <failure modes found in testing>
Monitoring:       <dashboards, alarms, on-call>
Last reviewed:    <date, reviewer>
```

Update the card in the same pull request as any change to the agent.

## 3. Controls
- [ ] Guardrail attached to every model the agent uses (content, denied topics, PII)
- [ ] Every tool with side effects sits behind AgentCore Gateway with a Cedar policy
- [ ] New policies rolled out in `LOG_ONLY` first, then `ENFORCE`
- [ ] Least-privilege IAM for the runtime, gateway and tool roles
- [ ] Inbound auth on every endpoint (OAuth / IAM — never `NONE` outside a demo)

## 4. Evaluation
- [ ] Built-in evaluators (goal success, correctness, helpfulness) on a fixed test set
- [ ] Custom evaluators for business rules (scope, tone, compliance)
- [ ] Online evaluation on a sample of live traffic
- [ ] Every production incident becomes a new test case

## 5. Audit trail
- [ ] Traces enabled (AgentCore Observability / OpenTelemetry)
- [ ] CloudWatch Logs retention set to match your regulatory requirement
- [ ] Long-term archive exported to S3 with Object Lock (write-once-read-many)
- [ ] You can answer: "show me everything the agent did for customer X on date Y"

## 6. Incident response
1. **Contain** — stop new requests: disable the AgentCore Runtime endpoint, set Lambda reserved concurrency to 0, or scale the ECS service to 0
2. **Investigate** — filter traces for errors since the alarm; check recent prompt, model, tool or dependency changes
3. **Fix** — add a Cedar rule, tighten the guardrail, roll back the prompt or model
4. **Communicate** — notify users and regulators within required windows (e.g. 72 hours under GDPR)
5. **Learn** — post-mortem, update the agent card, add the case to the evaluation set
