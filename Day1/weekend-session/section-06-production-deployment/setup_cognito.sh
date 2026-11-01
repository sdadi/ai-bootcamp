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
# The first run creates the pool and saves its details to .env.cognito in the
# current folder (git-ignored). Running it again — in the same or a new terminal,
# from the same folder — reuses that pool and just fetches a fresh token, so you
# never end up with a second pool. Delete the pool and .env.cognito to clean up.
#
# Exports: POOL_ID, CLIENT_ID, DISCOVERY_URL, BEARER_TOKEN, AUTH_CONFIG

_cognito_setup() {
  if [ -f .env.cognito ]; then
    # Reuse the pool created earlier
    set -a; . ./.env.cognito; set +a
    if ! aws cognito-idp describe-user-pool --user-pool-id "$POOL_ID" --region "$REGION" > /dev/null 2>&1; then
      echo "Error: can't reach the user pool saved in .env.cognito ($POOL_ID)." >&2
      echo "       Check your AWS login (aws sts get-caller-identity). If the pool was deleted, remove .env.cognito and run this again." >&2
      return 1
    fi
  else
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

    # App client (public client, username/password auth). Access tokens last 1 day
    # (Cognito's maximum; the default is 1 hour) because the deployed agents call each
    # other with the token they were deployed with — see the multi-agent README.
    CLIENT_ID=$(aws cognito-idp create-user-pool-client \
      --user-pool-id "$POOL_ID" \
      --client-name "AgentCoreClient" \
      --no-generate-secret \
      --explicit-auth-flows "ALLOW_USER_PASSWORD_AUTH" "ALLOW_REFRESH_TOKEN_AUTH" \
      --access-token-validity 1 --token-validity-units AccessToken=days \
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

    # Save what later runs need (git-ignored; holds the test user's password)
    printf "%s\n" "REGION='$REGION'" "COGNITO_USERNAME='$COGNITO_USERNAME'" "COGNITO_PASSWORD='$COGNITO_PASSWORD'" \
      "POOL_ID='$POOL_ID'" "CLIENT_ID='$CLIENT_ID'" "DISCOVERY_URL='$DISCOVERY_URL'" > .env.cognito
    chmod 600 .env.cognito
  fi

  export REGION POOL_ID CLIENT_ID DISCOVERY_URL
  export AUTH_CONFIG="{\"customJWTAuthorizer\":{\"discoveryUrl\":\"$DISCOVERY_URL\",\"allowedClients\":[\"$CLIENT_ID\"]}}"
  cognito_refresh_token
}

# Get a fresh access token (it lasts 1 day). Call this again any time.
cognito_refresh_token() {
  BEARER_TOKEN=$(aws cognito-idp initiate-auth \
    --client-id "$CLIENT_ID" \
    --auth-flow USER_PASSWORD_AUTH \
    --auth-parameters "USERNAME=$COGNITO_USERNAME,PASSWORD=$COGNITO_PASSWORD" \
    --region "$REGION" --query 'AuthenticationResult.AccessToken' --output text)
  [ -n "$BEARER_TOKEN" ] && [ "$BEARER_TOKEN" != "None" ] || { echo "Error: could not get a token." >&2; return 1; }
  export BEARER_TOKEN

  echo ""
  echo "Pool ID:       $POOL_ID   (saved in .env.cognito)"
  echo "Client ID:     $CLIENT_ID"
  echo "Discovery URL: $DISCOVERY_URL"
  echo "Bearer Token:  ${BEARER_TOKEN:0:25}...  (exported as \$BEARER_TOKEN, valid for 1 day)"
  echo "Auth config:   exported as \$AUTH_CONFIG"
  echo ""
}

_cognito_setup
