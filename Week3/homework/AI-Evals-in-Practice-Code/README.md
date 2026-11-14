# AI Evals in Practice: Code

**Author: Shivank Agrawal** · Future with Shivank

Runnable code for the 5 chapters of the *AI Evals in Practice* notes. Every script evaluates
**DoubtBot**, an AI teaching assistant for a beginner Python DSA course. Run the scripts in order;
each one is short, fully commented, and ends with an example of its output.

## Setup (once)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."                          # Windows PowerShell: $env:OPENAI_API_KEY="sk-..."
```

Python 3.10 or newer. Run every command from this folder.

## Models used

| Role | Model | Why |
|---|---|---|
| The product (DoubtBot, agents) | `gpt-6.1-terra` | strong, affordable |
| The judge | `gpt-6-luna` | small, cheap, fast: you grade thousands of replies |
| Test-data generator, simulated user | `gpt-6-astra` | a different, stronger model than the product |
| Retrieval | `text-embedding-3-small` | embeddings for the course notes |

Three API rules the code follows:
1. `temperature` only works with reasoning switched off: `reasoning_effort="none"` (Terra and Luna). Astra always reasons, so it never gets a temperature.
2. Tool calling on Terra needs the **Responses API** (`client.responses.create`). Tools there are strict by default, so each tool schema has `"additionalProperties": False`.
3. Structured replies use `client.chat.completions.parse(response_format=PydanticModel)` and are read from `.choices[0].message.parsed`.

## Run order

| Script | Notes section | What it shows | Needs key |
|---|---|---|---|
| `python doubtbot.py` | Ch 1 | The product under test: retrieve → answer → escalate | ✅ |
| `python ch1_01_first_eval.py` | 1.1 | Your first eval: test set + marking rule + score | ✅ |
| `python ch1_02_contamination_check.py` | 1.2 | Spot benchmark items the model has memorised | ✅ |
| `python ch1_03_product_vs_model.py` | 1.3 | Same model, two prompts = two different products | ✅ |
| `python ch1_04_judge_position_bias.py` | 1.5 | A judge that changes its mind when you swap the order | ✅ |
| `python ch2_01_rule_based_checks.py` | 2.3 | Regex checks: exact answer + PII guardrail | – |
| `python ch2_02_reference_metrics.py` | 2.3 | ROUGE-1 and embedding similarity, and how both get fooled | ✅ |
| `python ch2_03_bootstrap_traces.py` | 2.7 | Replay old forum questions to get traces before launch | ✅ |
| `python ch2_04_error_taxonomy.py` | 2.6 | Turn TA notes into an error taxonomy (Structured Outputs) | ✅ |
| `python ch3_01_precision_at_k.py` | 3.3 | Component metric: is retrieval fetching the right notes? | ✅ |
| `python ch3_02_llm_judge.py` | 3.6 | The LLM judge: accuracy (1-5) and integrity (PASS/FAIL) | ✅ |
| `python ch3_03_judge_calibration.py` | 3.7 | Judge vs TA labels: agreement, TPR, TNR, Cohen's kappa | ✅ |
| `python ch3_04_testset_from_history.py` | 3.9 | Test cases from forum history (silver test set) | ✅ |
| `python ch3_05_testset_synthetic.py` | 3.9 | Synthetic test messages with a stronger model | ✅ |
| `python ch3_06_test_set_size.py` | 3.10 | How many test cases you need (4 formulas) | – |
| `python ch3_07_release_gate.py` | 3.8, 3.11 | Golden set → product → evaluators → SHIP / NO SHIP | ✅ |
| `python ch4_01_pilot_ab_test.py` | 4.1 | Pilot A/B: is the gain real or luck? (two-proportion z-test) | – |
| `python ch4_02_nightly_monitoring.py` | 4.5 | Judge a random slice of live traffic, alert, review queue | ✅ |
| `pytest -q ch5_01_test_doubtbot_evals.py` | 5.2 | Guardrails as pytest tests for CI | ✅ |
| `python ch5_02_fairness_check.py` | 5.3 | Governance as a test: same question, different names | ✅ |
| `python ch5_03_multi_turn_n_plus_one.py` | 5.4 | N + 1: does escalation survive long chats? | ✅ |
| `python ch5_04_word_error_rate.py` | 5.5 | Word error rate for a voice mode | – |
| `python ch5_05_implied_meaning_judge.py` | 5.6 | A "sounds human" rubric: implied meaning, 1-5 | ✅ |
| `python ch5_06_redact_and_topics.py` | 5.7 | Redact PII, then classify topics for analytics | ✅ |
| `python ch5_07_simulated_user_agent.py` | 5.8 | Mini τ²-bench: agent + simulated student + graded end state | ✅ |

Scripts marked – need no API key. A full run of everything costs well under a dollar.

## Folder layout

```
doubtbot.py          the product we evaluate (used by many scripts)
judges.py            judge() + rubrics from Chapter 3 (used by later scripts)
ch1_ ... ch5_        one script per worked example, in chapter order
data/
  course_notes.json       DoubtBot's knowledge (includes 2 planted real-world problems: a JVM
                          "heap" wiki page and a mentor roster with phone numbers)
  test_doubts.json        ch1_03 · student doubts tagged concept / assignment
  comparison_pairs.json   ch1_04 · answer pairs of similar quality
  forum_threads.json      ch2_03, ch3_04 · old threads with the accepted TA reply
  open_coding_notes.txt   ch2_04 · TA notes from reading traces
  retrieval_labels.json   ch3_01 · which notes are relevant for each question
  calibration_set.json    ch3_03 · replies labelled PASS/FAIL by TAs
  golden_set.jsonl        ch3_07, ch5_01 · the release test set (4 slices)
  live_traces.json        ch4_02 · yesterday's production traces
  multi_turn_chats.json   ch5_03 · short and long chats ending in distress
  first_messages.json     ch5_06 · first messages of the week
  course_admin/           ch5_07 · policy.md, db.json, task.json for the agent simulation
outputs/             created when you run the scripts (traces, test sets, release run)
```

## Teaching tips

- LLM outputs vary, so your numbers will not match the example outputs exactly. That is the point of
  Chapter 3: measure on enough cases, and calibrate the judge before you trust it.
- `data/course_notes.json` contains an internal mentor roster with phone numbers that was indexed by
  mistake. The privacy slice in `ch3_07` and `ch5_01` attacks it; if any run leaks a number, the gate
  turns red. The prompt rule alone is not a reliable fix: delete the `ops_batch_roster` entry and re-run.
- To see a change in quality, edit the `SYSTEM` prompt in `doubtbot.py` and re-run `ch3_07` (the release gate).
