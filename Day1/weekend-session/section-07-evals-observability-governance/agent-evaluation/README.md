# End-to-End Agent Evaluation (AgentCore Evaluations)

| Step | What happens |
|---|---|
| 1 | Deploy a travel assistant to AgentCore Runtime (CodeBuild — no Docker) |
| 2 | Create a custom LLM-as-judge evaluator (5-point scale) |
| 3 | Invoke the agent with in-scope and out-of-scope prompts in one session |
| 4 | Run built-in evaluators: GoalSuccessRate (session), Correctness and Helpfulness (trace) |
| 5 | Run the custom evaluator and see it catch the out-of-scope answer |
| 6 | Print a score table, save results to JSON, clean up |

## The agent

`agent_app.py` has three tools — `get_flight_info`, `get_hotel_recommendations`, `get_weather_forecast` — and is told to stay in the travel domain. The custom evaluator (`travel_quality_metric.json`) scores any out-of-scope answer as **Very Poor (0.0)**.

## Prerequisites

- IAM: `bedrock-agentcore:*`, `ecr:*`, `iam:CreateRole`/`PassRole`, `codebuild:*`, `s3:*`, `logs:*`, `xray:*`
- CloudWatch Transaction Search enabled

## Run

```bash
pip install -r requirements.txt
jupyter lab end_to_end_eval.ipynb
```

First deployment takes 5–10 minutes. Expected: travel prompts score **Good / Very Good**; the Python-scripting prompt scores **Very Poor** on the custom evaluator.

## Groq edition (no Bedrock models)

`end_to_end_eval_groq.ipynb` keeps the same lifecycle — deploy to AgentCore Runtime, invoke one session, evaluate, summarise, clean up — with two changes:

- **The agent** (`agent_app_groq.py`, container packages in `agent_requirements-groq.txt`) runs on Groq; the launch passes `GROQ_API_KEY` to the runtime.
- **The evaluators** come from the open-source [Strands Evals](https://github.com/strands-agents/evals) SDK with a **Groq judge** (`GROQ_JUDGE_MODEL_ID`, default `openai/gpt-oss-120b`), because AgentCore Evaluations judges only with Bedrock models: `GoalSuccessRateEvaluator`, `CorrectnessEvaluator`, `HelpfulnessEvaluator`, plus the custom travel judge built from `travel_quality_metric.json`. The traces are read from the runtime's CloudWatch log group, as before.

```bash
pip install -r requirements-groq.txt
jupyter lab end_to_end_eval_groq.ipynb
```

- **Transaction Search** must be ACTIVE before you invoke the agent, or the traces are dropped. The launch turns it on, but it can take ~7 minutes the first time; the notebook waits for it.
- **What to expect:** gpt-oss usually *declines* the out-of-scope Python question. The built-in evaluators don't know the agent's scope and mark that refusal as unhelpful; the custom judge (its rubric gets one extra line: a polite refusal is correct) scores it Very Good, and would score an out-of-scope *answer* Very Poor.
- **Cost and time:** about 9 minutes end to end; the judge makes ~7 calls (~10k tokens), well inside Groq's free tier. `destroy` leaves the runtime's log group, the CloudWatch traces delivery and the toolkit's S3 bucket — delete them in the console if you don't need them.
