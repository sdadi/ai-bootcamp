# ECS Fargate Deployment: Hospital Scheduling Agent

A Strands agent served by FastAPI on ECS Fargate behind an Application Load Balancer. It coordinates surgical scheduling by checking provider availability, equipment and operating rooms (mock tools), and serves a small chat UI.

```
Browser (chat UI) → ALB (port 80) → ECS Fargate task → Amazon Bedrock
                                       │
                                 FastAPI + Strands agent
                                 (tools: provider calendar,
                                  equipment check, booking)
```

## Prerequisites

- AWS credentials **in the terminal you deploy from** (`aws sts get-caller-identity` must print your account). With SSO: `aws sso login --profile <p>`, then `set -a; source ../../.env; set +a` (or `export AWS_PROFILE=<p>`)
- A model you can call: if Claude isn't enabled in your account yet, `export BEDROCK_MODEL_ID=us.amazon.nova-2-lite-v1:0` before deploying (see "Before you start" in the section README)
- Docker running (`docker info`)
- AWS CLI v2, ideally with the pager off: `export AWS_PAGER=""`
- A default VPC with at least two public subnets. New accounts have one; if yours doesn't (`aws ec2 describe-vpcs --filters Name=isDefault,Values=true` prints nothing), create it: `aws ec2 create-default-vpc --region us-east-1`

## Run locally first (optional)

```bash
pip install -r requirements.txt
cd app && uvicorn app:app --port 8080
# open http://localhost:8080
```

## Deploy

```bash
./deploy.sh          # or: bash deploy.sh
```

Creates: ECR repo and image, IAM roles, ECS cluster, task definition, ALB, target group, security groups, CloudWatch log group, and the Fargate service. On Windows, run from WSL or Git Bash.

- **How long:** about 3–6 minutes. The image is built for `linux/amd64`, so on an Apple Silicon Mac the first build runs under emulation and is the slow part (later builds are cached).
- **The model:** `deploy.sh` passes `BEDROCK_MODEL_ID` to the task as `MODEL_ID`. To switch models, export the new value and run `./deploy.sh` again — it's safe to repeat and rolls the service onto a new task-definition revision.
- **Who can reach it:** by default the load balancer only accepts traffic from **your current public IP** (detected via checkip.amazonaws.com). To open it to everyone for a live class demo:

  ```bash
  ALB_INGRESS_CIDR=0.0.0.0/0 ./deploy.sh
  ```

## Test

`deploy.sh` prints the URL as soon as it has created the load balancer — **before the load balancer is ready**. For the first 2–4 minutes `curl` says `Could not resolve host`, then you may get a `503` while the task starts. Wait for it:

```bash
aws elbv2 wait load-balancer-available --names hospital-scheduling-agent-alb
curl http://<ALB_DNS>/health                        # {"status":"healthy"} when ready
```

Then open the URL in a browser (the chat UI), or:

```bash
curl -X POST http://<ALB_DNS>/schedule -H 'Content-Type: application/json' \
  -d '{"message": "Schedule a knee arthroscopy for patient P-1234 with Dr. Smith on 2026-11-18"}'
```

A good answer comes after three tool calls (provider availability, equipment, booking) and contains a booking reference, an operating room and a time slot. The chat UI sends the conversation so far with each message (`history`), so a follow-up like "What was my booking reference?" is answered from the first turn.

- **Use explicit dates.** The agent has no calendar tool, so "next Tuesday" (in the example `deploy.sh` prints) or "next Wednesday" (a suggestion button in the UI) makes it ask which date you mean. The other suggestion button uses 2026-03-18, a date in the past.
- **Logs:** `aws logs tail /ecs/scheduling-agent --follow --region us-east-1` — tool calls and errors show up here.

## Clean up

```bash
./cleanup.sh
```

It takes about 8 minutes: it scales the service to zero and waits for it to drain (the target group keeps connections open for 300 s), then deletes the load balancer, target group, cluster, ECR repository, IAM roles, log group and security groups.

- If it stops at `(END)`, that's the AWS CLI pager showing a long output — press `q` and it continues (or run it with `AWS_PAGER="" ./cleanup.sh`).
- It deregisters the task definitions but doesn't delete them, so they stay listed as INACTIVE. They cost nothing; to remove them:
  ```bash
  aws ecs list-task-definitions --family-prefix scheduling-agent --status INACTIVE --query 'taskDefinitionArns' --output text \
    | xargs -n 10 aws ecs delete-task-definitions --output text --query 'failures' --task-definitions
  ```
- If you created the default VPC for this lab, it's free to keep.

## Cost

About $0.05/hour for one 1 vCPU / 2 GB task, plus the ALB's hourly charge. Run `./cleanup.sh` when you finish.

## Groq edition (no Bedrock models)

| File | Difference from the Bedrock version |
|---|---|
| `app/app_groq.py`, `requirements-groq.txt`, `Dockerfile.groq` | The agent's model runs on Groq (`openai/gpt-oss-120b` by default) |
| `deploy-groq.sh` | `-groq` resource names (it can run next to the Bedrock deployment); stores `GROQ_API_KEY` in Secrets Manager (`hospital-agent-groq/groq-api-key`) and injects it through the task definition's `secrets`, so the key never appears in the task definition; no Bedrock task role |
| `cleanup-groq.sh` | Also deletes the secret, and removes deregistered task-definition revisions |

```bash
export GROQ_API_KEY=...            # or: set -a; source ../../.env; set +a
./deploy-groq.sh                   # ./cleanup-groq.sh when done

# local test
pip install -r requirements-groq.txt && cd app && uvicorn app_groq:app --port 8080
```

- **Default VPC required.** `deploy-groq.sh` checks for one first. If your account has none: `aws ec2 create-default-vpc --region us-east-1`.
- **Timings:** on an Apple Silicon Mac the `linux/amd64` image is built under emulation, so the first build takes ~10 minutes; later builds are cached. If `docker push` fails on a flaky connection, just re-run `./deploy-groq.sh` — it is safe to repeat. The task then takes 2–3 minutes to pass health checks.
