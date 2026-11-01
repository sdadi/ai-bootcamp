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

## Running it

Run the cells in order; the whole notebook takes about a minute. The kernel needs your AWS profile (`AWS_PROFILE` in `.env` or your environment) and, if Claude isn't enabled in your account yet, `BEDROCK_MODEL_ID=us.amazon.nova-2-lite-v1:0` — see "Before you start" in the section README.

What you should see:

- Step 1: `Spec uploaded: s3://gateway-specs-<account>-<region>/weather_openapi.json`
- Steps 2–3: the credential provider ARN and the role ARN
- Step 4: `Gateway status: CREATING` → `READY`, then the gateway URL
- Step 5: `Target status: READY`
- Step 6: `Tools exposed by the gateway: ['WeatherAPI___getForecast']`, one `Tool #N: WeatherAPI___getForecast` line per call, and a forecast with **real numbers** (daily highs/lows, rain in mm) for the cities you asked about
- Step 7: `Deleted gateway ...`, `Deleted credential provider ...`, `Deleted role ...`, `Deleted bucket ...` — nothing is left behind

**Why the key is sent as `X-Demo-Key`.** OpenAPI targets require an outbound credential, so the notebook stores a placeholder key. Open-Meteo ignores most extra headers, but it redirects any request carrying an `X-Api-Key` header to its paid API (`303 → customer-api.open-meteo.com`), and the gateway can't follow that redirect. With `X-Api-Key`, every tool call fails and the agent answers with something like "there was a technical issue retrieving the weather data". Both notebooks use `X-Demo-Key`.

## Troubleshooting

| You see | What is happening | Fix |
|---|---|---|
| `FileNotFoundError: ... 'weather_openapi.json'` in Step 1 | The kernel isn't running in this folder (in VS Code, `jupyter.notebookFileRoot` is set to `${workspaceFolder}`) | Set it to `${fileDirname}` and restart the kernel — don't change the path in the cell |
| The agent says it couldn't retrieve the weather, or the tool result contains `OpenAPIClientException` | The gateway's call to Open-Meteo failed; with an `X-Api-Key` header it's the redirect described above | Check Step 5 uses `"credentialParameterName": "X-Demo-Key"` |
| The tool returns `An internal error occurred. Please retry later.` | Open-Meteo rejected a parameter — some models invent `daily` variable names (e.g. `maxtemp_c`) | Ask again, or name the variables in your question (`temperature_2m_max`, `precipitation_sum`) |
| `Model use case details have not been submitted` / `ExpiredToken` | Model access or AWS login | See the section README's troubleshooting table |

**If you stop halfway**, still run the Step 7 cell — it only needs the variables from the earlier cells. If the kernel was restarted and they're gone, delete by name: the gateway `weather-gateway-<suffix>` (its target first), the API-key credential provider `weather-api-key-<suffix>`, the IAM role `AgentCoreGatewayDemoRole-<suffix>`, and the bucket `gateway-specs-<account>-<region>`. The bucket name is the same on every run, so two people running the notebook in one account at the same time share it.

## Groq edition

`gateway_openapi_target_groq.ipynb` is the same lab with the agent's model on Groq (`GROQ_API_KEY` in your environment or `.env`). The gateway, AgentCore Identity, S3 and IAM are unchanged and still need AWS credentials — no Bedrock model access.
