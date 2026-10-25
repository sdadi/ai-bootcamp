#!/bin/bash
# Create a Cognito user pool + app client + user, and get an OAuth access token
# for calling agents deployed on AgentCore Runtime with JWT (OAuth) auth.
#
# Prerequisites: AWS CLI v2
#   export REGION=us-east-1
#   export COGNITO_USERNAME=testuser
#   export COGNITO_PASSWORD='YourPassword123!'
#
# Usage (note: "source", so the variables stay in your shell):
#   source setup_cognito.sh
#
# Exports: POOL_ID, CLIENT_ID, DISCOVERY_URL, BEARER_TOKEN

_cognito_setup() {
  if [ -z "$REGION" ] || [ -z "$COGNITO_USERNAME" ] || [ -z "$COGNITO_PASSWORD" ]; then
    echo "Error: set REGION, COGNITO_USERNAME and COGNITO_PASSWORD first." >&2
    return 1
  fi

  # User pool
  POOL_ID=$(aws cognito-idp create-user-pool \
    --pool-name "AgentCoreUserPool" \
    --policies '{"PasswordPolicy":{"MinimumLength":8}}' \
    --region "$REGION" --query 'UserPool.Id' --output text)
  [ -n "$POOL_ID" ] && [ "$POOL_ID" != "None" ] || { echo "Error: could not create the user pool." >&2; return 1; }

  # App client (public client, username/password auth)
  CLIENT_ID=$(aws cognito-idp create-user-pool-client \
    --user-pool-id "$POOL_ID" \
    --client-name "AgentCoreClient" \
    --no-generate-secret \
    --explicit-auth-flows "ALLOW_USER_PASSWORD_AUTH" "ALLOW_REFRESH_TOKEN_AUTH" \
    --region "$REGION" --query 'UserPoolClient.ClientId' --output text)
  [ -n "$CLIENT_ID" ] && [ "$CLIENT_ID" != "None" ] || { echo "Error: could not create the app client." >&2; return 1; }

  # User with a permanent password
  aws cognito-idp admin-create-user \
    --user-pool-id "$POOL_ID" --username "$COGNITO_USERNAME" \
    --region "$REGION" --message-action SUPPRESS > /dev/null || return 1
  aws cognito-idp admin-set-user-password \
    --user-pool-id "$POOL_ID" --username "$COGNITO_USERNAME" --password "$COGNITO_PASSWORD" \
    --region "$REGION" --permanent > /dev/null || return 1

  DISCOVERY_URL="https://cognito-idp.$REGION.amazonaws.com/$POOL_ID/.well-known/openid-configuration"
  export POOL_ID CLIENT_ID DISCOVERY_URL
  cognito_refresh_token
}

# Get a fresh access token (they expire after 1 hour). Call this again any time.
cognito_refresh_token() {
  BEARER_TOKEN=$(aws cognito-idp initiate-auth \
    --client-id "$CLIENT_ID" \
    --auth-flow USER_PASSWORD_AUTH \
    --auth-parameters "USERNAME=$COGNITO_USERNAME,PASSWORD=$COGNITO_PASSWORD" \
    --region "$REGION" --query 'AuthenticationResult.AccessToken' --output text)
  export BEARER_TOKEN

  echo ""
  echo "Pool ID:       $POOL_ID"
  echo "Client ID:     $CLIENT_ID"
  echo "Discovery URL: $DISCOVERY_URL"
  echo "Bearer Token:  ${BEARER_TOKEN:0:25}...  (exported as \$BEARER_TOKEN)"
  echo ""
  echo "Token expires in 1 hour. Refresh with:  cognito_refresh_token"
}

_cognito_setup
