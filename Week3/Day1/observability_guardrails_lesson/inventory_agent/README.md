# Inventory Management Agent: built step by step
*by Shivank Agarwal*

An AI assistant that answers questions like **"Should I reorder toothpaste?"** by reading a live
inventory database. It's built in **8 small versions**, and each version adds **one** idea.
Run them in order to watch a single prompt grow into a production-style agent.

| Version | What it adds | Idea it teaches |
|---|---|---|
| `inventoryplanner_v0.py` | One prompt, numbers typed in by hand | The simplest LLM call |
| `inventoryplanner_v1.py` | A Gradio web form | Putting a UI on a model |
| `inventoryplanner_v2.py` | SQLite database + tools + agent (chat UI) | RAG and tool calling |
| `inventoryplanner_v3.py` | Rule-based input guardrails | Blocking prompt injection & jailbreaks |
| `inventoryplanner_v4.py` | Answer cache with 5-minute expiry + stats | Speed and cost savings |
| `inventoryplanner_v5.py` | Send an email alert | Write actions (the agent can act, not just read) |
| `inventoryplanner_v6.py` | LangSmith tracing | Observability |
| `inventoryplanner_v7.py` | `calculate_reorder_quantity` tool | Let code do the maths, not the model |

`database.py` creates `inventory.db` (19 sample products) automatically the first time v2+ runs.

## How one question flows (v7)

```
user question
   -> guardrail check        (blocked? reply with a warning)
   -> cache check            (asked in the last 5 minutes? reply instantly with 🔄)
   -> agent (gpt-6-luna)     picks a tool:
        get_inventory_data / get_all_inventory / calculate_reorder_quantity
   -> tool reads inventory.db, returns facts
   -> agent writes the answer
   -> answer saved in the cache, shown in the chat, trace sent to LangSmith
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then put your keys in .env
python inventoryplanner_v0.py
python inventoryplanner_v7.py   # opens a web page at http://127.0.0.1:7860
```

- **Every version** needs `OPENAI_API_KEY`.
- **v5–v7** email button needs the `EMAIL_*` settings (for Gmail: an App Password, not your normal password).
- **v6–v7** tracing needs the `LANGSMITH_*` settings (free account at smith.langchain.com).
  Without them the app still works; it just doesn't send traces.

## The reorder rule used everywhere

```
days remaining = current stock / average daily demand
if days remaining <= lead time  ->  reorder at least (average daily demand x lead time) units
```
Example: toothpaste, 20 in stock, 5 sold per day, 4-day delivery -> 4 days left = lead time -> reorder 20 units now.
