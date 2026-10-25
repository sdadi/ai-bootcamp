# AgentCore Gateway — REST API → MCP Tools

`gateway_openapi_target.ipynb` turns a REST API into agent tools with zero tool code:

1. Upload `weather_openapi.json` (an OpenAPI spec for the free Open-Meteo API) to S3
2. Store the API's credential in AgentCore Identity
3. Create a gateway IAM role and an MCP gateway
4. Register the API as a target straight from the S3 spec → the gateway creates the `WeatherAPI___getForecast` tool
5. Connect a Strands agent to the gateway URL and ask for a forecast
6. Clean up everything

To use your own API, replace `weather_openapi.json` with your spec, and put your real API key (or an OAuth provider) in Step 2.

The demo gateway uses `authorizerType="NONE"` and is deleted at the end. For anything long-lived use `CUSTOM_JWT` (OAuth) or `AWS_IAM`.

## Groq edition

`gateway_openapi_target_groq.ipynb` is the same lab with the agent's model on Groq (`GROQ_API_KEY` in your environment or `.env`). The gateway, AgentCore Identity, S3 and IAM are unchanged and still need AWS credentials — no Bedrock model access.

It sends the placeholder credential as `X-Demo-Key`: Open-Meteo now redirects any request carrying an `X-Api-Key` header to its paid API, and the gateway can't follow that redirect, so every tool call would fail.
