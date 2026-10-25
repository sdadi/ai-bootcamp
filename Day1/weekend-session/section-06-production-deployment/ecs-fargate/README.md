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

- AWS credentials (`aws sts get-caller-identity`)
- Docker running (`docker info`)
- AWS CLI v2
- A default VPC with at least two public subnets (new accounts have one)

## Run locally first (optional)

```bash
pip install -r requirements.txt
cd app && uvicorn app:app --port 8080
# open http://localhost:8080
```

## Deploy

```bash
chmod +x deploy.sh cleanup.sh
./deploy.sh
```

Creates: ECR repo and image, IAM roles, ECS cluster, task definition, ALB, target group, security groups, CloudWatch log group, and the Fargate service. Takes 3–4 minutes; the task needs another 2–3 minutes to pass health checks. On Windows, run from WSL or Git Bash.

By default the load balancer only accepts traffic from **your current public IP** (detected via checkip.amazonaws.com). To open it to everyone for a live class demo:

```bash
ALB_INGRESS_CIDR=0.0.0.0/0 ./deploy.sh
```

## Test

Open the ALB URL printed at the end, or:

```bash
curl http://<ALB_DNS>/health
curl -X POST http://<ALB_DNS>/schedule -H 'Content-Type: application/json' \
  -d '{"message": "Schedule a knee arthroscopy for patient P-1234 with Dr. Smith on 2026-11-18"}'
```

Logs: `aws logs tail /ecs/scheduling-agent --follow --region us-east-1`

## Clean up

```bash
./cleanup.sh
```

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
