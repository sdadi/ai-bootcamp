# Memory and MCP: code for the two lectures

By Shivank Agarwal. Every file uses OpenAI `gpt-6-luna`.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then paste your OpenAI key into .env
```

The notebooks install what they need in their first cell and ask for the key if it isn't already set,
so they also run as-is in Google Colab.

## Files

| File | Lecture | What it shows |
|---|---|---|
| `short_term_memory.ipynb` | Memory | Conversation buffer, then three ways to keep it small: trimming, deleting, summarising |
| `memory_demo_1.py` | Memory | Semantic, episodic and procedural memory feeding one answer (a LangGraph pipeline) |
| `memory_demo_2.py` | Memory | The agent updates a customer's preferences mid-conversation and saves them to `episodic_memory.json` |
| `fastmcp_demo_v1.ipynb` | MCP | An MCP server built with FastMCP (time + weather tools) used by `gpt-6-luna` through function calling |

Run the scripts with `python memory_demo_1.py` and `python memory_demo_2.py`.
`memory_demo_2.py` creates `episodic_memory.json` next to itself: open it after the run to see the saved memory.
