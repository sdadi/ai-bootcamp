# Part 2 helper — every step in the README starts with:   source session.sh
#
# - Sources ../setup_cognito.sh: the first time it creates the Cognito pool (needs
#   REGION, COGNITO_USERNAME, COGNITO_PASSWORD); after that it reuses the pool saved
#   in .env.cognito. Either way AUTH_CONFIG and a fresh BEARER_TOKEN are exported.
# - Defines runtime_url NAME [mcp]: the invocation URL of a deployed runtime, looked
#   up by name; it fails with a message if that runtime isn't deployed.
#
# It returns non-zero on any problem, so "source session.sh && ..." stops right there.

if [ ! -f stock_a2a_agent.py ] || [ ! -f ../setup_cognito.sh ]; then
  echo "Error: run this from the multi-agent-on-agentcore folder." >&2
  return 1
fi

export REGION="${REGION:-${AWS_REGION:-us-east-1}}"

runtime_url() {
  local arn enc
  arn=$(aws bedrock-agentcore-control list-agent-runtimes --region "$REGION" \
    --query "agentRuntimes[?agentRuntimeName=='$1'].agentRuntimeArn" --output text)
  if [ -z "$arn" ] || [ "$arn" = "None" ]; then
    echo "Error: no AgentCore runtime named '$1' — deploy it first." >&2
    return 1
  fi
  enc=$(printf %s "$arn" | jq -sRr '@uri')
  if [ "$2" = "mcp" ]; then
    echo "https://bedrock-agentcore.$REGION.amazonaws.com/runtimes/$enc/invocations?qualifier=DEFAULT"
  else
    echo "https://bedrock-agentcore.$REGION.amazonaws.com/runtimes/$enc/invocations/"
  fi
}

source ../setup_cognito.sh
