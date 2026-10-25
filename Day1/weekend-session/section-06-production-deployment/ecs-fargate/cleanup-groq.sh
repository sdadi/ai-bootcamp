#!/usr/bin/env bash
set -euo pipefail

# ── Configuration (must match deploy-groq.sh) ──────────────────────
APP_NAME="hospital-agent-groq"
CLUSTER_NAME="agent-cluster-groq"
SERVICE_NAME="scheduling-agent-groq-service"
TASK_FAMILY="scheduling-agent-groq"
SECRET_NAME="${APP_NAME}/groq-api-key"
EXEC_ROLE_NAME="${APP_NAME}-exec-role"
ALB_NAME="${APP_NAME}-alb"
TG_NAME="${APP_NAME}-tg"
ALB_SG_NAME="${APP_NAME}-alb-sg"
ECS_SG_NAME="${APP_NAME}-ecs-sg"

REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-$(aws configure get region 2>/dev/null || true)}}"
REGION="${REGION:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo "=== Cleaning up Hospital Scheduling Agent (Groq) ==="
echo ""
FAILED=()

# ── Delete ECS service ─────────────────────────────────────────────
echo "Deleting ECS service..."
aws ecs update-service --cluster "${CLUSTER_NAME}" --service "${SERVICE_NAME}" \
  --desired-count 0 --region "${REGION}" --output text 2>/dev/null || true
aws ecs delete-service --cluster "${CLUSTER_NAME}" --service "${SERVICE_NAME}" \
  --force --region "${REGION}" --output text 2>/dev/null || true
echo "  Waiting for the service to become inactive (up to ~10 min)..."
aws ecs wait services-inactive --cluster "${CLUSTER_NAME}" --services "${SERVICE_NAME}" \
  --region "${REGION}" 2>/dev/null || true

# ── Delete ALB resources ──────────────────────────────────────────
echo "Deleting ALB resources..."
ALB_ARN=$(aws elbv2 describe-load-balancers --names "${ALB_NAME}" \
  --region "${REGION}" --query 'LoadBalancers[0].LoadBalancerArn' --output text 2>/dev/null || true)
if [ -n "${ALB_ARN}" ] && [ "${ALB_ARN}" != "None" ]; then
  # Delete listeners first
  LISTENERS=$(aws elbv2 describe-listeners --load-balancer-arn "${ALB_ARN}" \
    --region "${REGION}" --query 'Listeners[*].ListenerArn' --output text 2>/dev/null || true)
  for arn in ${LISTENERS}; do
    aws elbv2 delete-listener --listener-arn "${arn}" --region "${REGION}" 2>/dev/null || true
  done
  aws elbv2 delete-load-balancer --load-balancer-arn "${ALB_ARN}" --region "${REGION}" 2>/dev/null || true
  echo "  Waiting for the load balancer to be deleted..."
  aws elbv2 wait load-balancers-deleted --load-balancer-arns "${ALB_ARN}" --region "${REGION}" 2>/dev/null \
    || FAILED+=("load balancer ${ALB_NAME}")
fi

TG_ARN=$(aws elbv2 describe-target-groups --names "${TG_NAME}" \
  --region "${REGION}" --query 'TargetGroups[0].TargetGroupArn' --output text 2>/dev/null || true)
if [ -n "${TG_ARN}" ] && [ "${TG_ARN}" != "None" ]; then
  aws elbv2 delete-target-group --target-group-arn "${TG_ARN}" --region "${REGION}" 2>/dev/null || true
fi
echo "  ALB and target group deleted."

# ── Deregister task definitions ────────────────────────────────────
echo "Deregistering task definitions..."
TASK_DEFS=$(aws ecs list-task-definitions --family-prefix "${TASK_FAMILY}" \
  --region "${REGION}" --query 'taskDefinitionArns' --output text 2>/dev/null || true)
for td in ${TASK_DEFS}; do
  aws ecs deregister-task-definition --task-definition "${td}" --region "${REGION}" --output text 2>/dev/null || true
done
# Deregistered revisions stay listed as INACTIVE until they are deleted
INACTIVE_DEFS=$(aws ecs list-task-definitions --family-prefix "${TASK_FAMILY}" --status INACTIVE \
  --region "${REGION}" --query 'taskDefinitionArns' --output text 2>/dev/null || true)
for td in ${INACTIVE_DEFS}; do
  aws ecs delete-task-definitions --task-definitions "${td}" --region "${REGION}" --output text >/dev/null 2>&1 || true
done

# ── Delete ECS cluster ─────────────────────────────────────────────
echo "Deleting ECS cluster..."
aws ecs delete-cluster --cluster "${CLUSTER_NAME}" --region "${REGION}" --output text >/dev/null 2>&1 \
  || FAILED+=("ECS cluster ${CLUSTER_NAME}")

# ── Delete ECR repository ─────────────────────────────────────────
echo "Deleting ECR repository..."
aws ecr delete-repository --repository-name "${APP_NAME}" --force --region "${REGION}" --output text 2>/dev/null || true

# ── Delete CloudWatch log group ────────────────────────────────────
echo "Deleting log group..."
aws logs delete-log-group --log-group-name "/ecs/${TASK_FAMILY}" --region "${REGION}" 2>/dev/null || true

# ── Delete security groups ─────────────────────────────────────────
echo "Deleting security groups..."
# Network interfaces can take a few minutes to be released after the ALB and
# tasks are gone, so retry each delete for up to ~5 minutes.

VPC_ID=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true \
  --region "${REGION}" --query 'Vpcs[0].VpcId' --output text)

for SG_NAME in "${ECS_SG_NAME}" "${ALB_SG_NAME}"; do
  SG_ID=$(aws ec2 describe-security-groups --filters Name=group-name,Values="${SG_NAME}" Name=vpc-id,Values="${VPC_ID}" \
    --region "${REGION}" --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)
  if [ -n "${SG_ID}" ] && [ "${SG_ID}" != "None" ]; then
    DELETED=false
    for attempt in $(seq 1 30); do
      if aws ec2 delete-security-group --group-id "${SG_ID}" --region "${REGION}" 2>/dev/null; then
        DELETED=true; break
      fi
      sleep 10
    done
    $DELETED || FAILED+=("security group ${SG_NAME} (${SG_ID})")
  fi
done

# ── Delete the Groq key secret ────────────────────────────────────
echo "Deleting the Groq API key secret..."
aws secretsmanager delete-secret --secret-id "${SECRET_NAME}" --force-delete-without-recovery \
  --region "${REGION}" --output text >/dev/null 2>&1 || true

# ── Delete IAM role ───────────────────────────────────────────────
echo "Deleting IAM role..."
aws iam delete-role-policy --role-name "${EXEC_ROLE_NAME}" --policy-name GroqSecretAccess 2>/dev/null || true
aws iam detach-role-policy --role-name "${EXEC_ROLE_NAME}" \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy 2>/dev/null || true
aws iam delete-role --role-name "${EXEC_ROLE_NAME}" 2>/dev/null || true
echo "  IAM role deleted."

echo ""
if [ "${#FAILED[@]}" -eq 0 ]; then
  echo "=== Cleanup complete ==="
  echo "All resources for ${APP_NAME} have been removed."
else
  echo "=== Cleanup finished with leftovers ==="
  printf '  - %s\n' "${FAILED[@]}"
  echo "Wait a few minutes and run ./cleanup-groq.sh again, or delete them in the AWS console."
  exit 1
fi
