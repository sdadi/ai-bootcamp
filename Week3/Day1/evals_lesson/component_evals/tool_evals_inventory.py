"""
Tool evals: does the agent pick the right calculator, the right number of times, at what cost?
By Shivank Agarwal. Model: OpenAI gpt-6-luna.

1. Six inventory tools (reorder point, EOQ, safety stock, days of supply, ABC class, today's date).
2. A tool-calling agent built with LangChain's create_agent.
3. 12 test scenarios with the tools we EXPECT: single-tool, multi-tool, ambiguous, and no-tool.
4. Metrics from the lecture's "Tool Use" row: precision, recall, calls per task, tool success rate,
   attribution, latency and cost per successful task.
"""
import math
import re
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

load_dotenv()                                        # reads OPENAI_API_KEY from .env

# gpt-6-luna list prices, USD per 1M tokens (openai.com/api/pricing, Nov 2026). Update if they change.
INPUT_PRICE_PER_1M = 0.10
OUTPUT_PRICE_PER_1M = 0.50
TOOL_COST_PER_CALL = 0.0                             # these tools run locally, so they're free; set a price for paid APIs


# ---------------------------------------------------------------------------
# 1. Tools
# ---------------------------------------------------------------------------
@tool
def reorder_point_calculator(avg_daily_demand: float, lead_time_days: float, safety_stock: float = 0) -> str:
    """Calculate reorder point: ROP = (avg_daily_demand * lead_time_days) + safety_stock."""
    rop = avg_daily_demand * lead_time_days + safety_stock
    return f"Reorder Point = {rop:.1f} units (demand {avg_daily_demand}/day × {lead_time_days} days lead time + {safety_stock} safety stock)"


@tool
def eoq_calculator(annual_demand: float, ordering_cost: float, holding_cost_per_unit: float) -> str:
    """Calculate Economic Order Quantity: EOQ = sqrt(2 * D * S / H)."""
    if holding_cost_per_unit <= 0:
        return "Error: holding cost must be > 0"
    eoq = math.sqrt(2 * annual_demand * ordering_cost / holding_cost_per_unit)
    return f"EOQ = {eoq:.1f} units (D={annual_demand}, S=${ordering_cost}, H=${holding_cost_per_unit})"


@tool
def safety_stock_calculator(z_score: float, demand_std_dev: float, lead_time_days: float) -> str:
    """Calculate safety stock: SS = Z * σ_demand * sqrt(lead_time)."""
    ss = z_score * demand_std_dev * math.sqrt(lead_time_days)
    return f"Safety Stock = {ss:.1f} units (Z={z_score}, σ={demand_std_dev}, LT={lead_time_days} days)"


@tool
def days_of_supply_calculator(on_hand_units: float, avg_daily_demand: float) -> str:
    """Calculate days of supply remaining: DoS = on_hand / avg_daily_demand."""
    if avg_daily_demand <= 0:
        return "Error: daily demand must be > 0"
    dos = on_hand_units / avg_daily_demand
    return f"Days of Supply = {dos:.1f} days ({on_hand_units} units ÷ {avg_daily_demand} units/day)"


@tool
def abc_classifier(annual_spend_pct: float) -> str:
    """Classify an inventory item as A, B, or C based on its share of total annual spend (0-100)."""
    if annual_spend_pct >= 70:
        cls, desc = "A", "high-value — tight control, frequent review"
    elif annual_spend_pct >= 20:
        cls, desc = "B", "medium-value — moderate review"
    else:
        cls, desc = "C", "low-value — simplified control (but check criticality)"
    return f"Class {cls}: {annual_spend_pct:.1f}% of annual spend — {desc}"


@tool
def inventory_date_info() -> str:
    """Return today's date for reorder scheduling and expiry calculations."""
    today = date.today()
    return f"Today is {today.strftime('%A, %B %d, %Y')} (week {today.isocalendar()[1]} of {today.year})"


tools = [reorder_point_calculator, eoq_calculator, safety_stock_calculator,
         days_of_supply_calculator, abc_classifier, inventory_date_info]


# ---------------------------------------------------------------------------
# 2. The agent under test
# ---------------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-6-luna", reasoning_effort="none")
agent = create_agent(
    llm,
    tools,
    system_prompt="""You are an inventory management assistant with access to calculation tools.
Use the appropriate tool(s) to answer inventory questions precisely.
Always use tools when they are relevant to the question.
Be precise in selecting the right tool for each task.""",
)


# ---------------------------------------------------------------------------
# 3. Test scenarios: the question, the tools we expect, and the ideal number of calls
# ---------------------------------------------------------------------------
test_scenarios = [
    # Single-tool
    {"question": "Calculate the reorder point for an item with 50 units/day demand, 7-day lead time, and 100 units safety stock.",
     "expected_tools": ["reorder_point_calculator"], "category": "single_tool", "requires_external_data": True, "optimal_tool_count": 1},
    {"question": "What is the EOQ for annual demand of 10000 units, ordering cost $50, holding cost $2 per unit?",
     "expected_tools": ["eoq_calculator"], "category": "single_tool", "requires_external_data": True, "optimal_tool_count": 1},
    {"question": "Calculate safety stock using Z=1.65, demand std dev of 20 units, and 5-day lead time.",
     "expected_tools": ["safety_stock_calculator"], "category": "single_tool", "requires_external_data": True, "optimal_tool_count": 1},
    {"question": "We have 800 units on hand and sell 40 units per day. How many days of supply do we have?",
     "expected_tools": ["days_of_supply_calculator"], "category": "single_tool", "requires_external_data": True, "optimal_tool_count": 1},
    {"question": "An item represents 5% of our annual spend. What ABC class is it?",
     "expected_tools": ["abc_classifier"], "category": "single_tool", "requires_external_data": True, "optimal_tool_count": 1},
    {"question": "What is today's date? I need to schedule a reorder.",
     "expected_tools": ["inventory_date_info"], "category": "single_tool", "requires_external_data": True, "optimal_tool_count": 1},

    # Multi-tool
    {"question": "Calculate the EOQ for D=5000, S=$30, H=$4, and also tell me today's date so I can schedule the first order.",
     "expected_tools": ["eoq_calculator", "inventory_date_info"], "category": "multi_tool", "requires_external_data": True, "optimal_tool_count": 2},
    {"question": "Our item has 30 units/day demand, 10-day lead time, no safety stock. Calculate the ROP and how many days 450 units on hand will last.",
     "expected_tools": ["reorder_point_calculator", "days_of_supply_calculator"], "category": "multi_tool", "requires_external_data": True, "optimal_tool_count": 2},

    # Ambiguous: the right tool is not named in the question
    {"question": "What is 2 * 5000 * 30 / 4 and then take the square root?",
     "expected_tools": ["eoq_calculator"], "category": "ambiguous",
     "acceptable_alternatives": [], "requires_external_data": True, "optimal_tool_count": 1},
    {"question": "Tell me about right now so I can timestamp my inventory report.",
     "expected_tools": ["inventory_date_info"], "category": "ambiguous", "requires_external_data": True, "optimal_tool_count": 1},

    # No tool needed: the model should answer from its own knowledge
    {"question": "What does FEFO stand for in warehouse management?",
     "expected_tools": [], "category": "no_tool", "requires_external_data": False, "optimal_tool_count": 0},
    {"question": "Explain the difference between cycle stock and safety stock.",
     "expected_tools": [], "category": "no_tool", "requires_external_data": False, "optimal_tool_count": 0},
]


# ---------------------------------------------------------------------------
# 4. What we record for every tool call and every task
# ---------------------------------------------------------------------------
@dataclass
class ToolCallLog:
    tool_name: str
    parameters: Dict[str, Any]
    output: str
    success: bool
    is_correct: bool


@dataclass
class TaskEvaluation:
    question: str
    category: str
    expected_tools: List[str]
    actual_tools: List[str]
    tool_call_logs: List[ToolCallLog]
    output: Optional[str]
    error: Optional[str]
    requires_external_data: bool
    optimal_tool_count: int
    acceptable_alternatives: List[List[str]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0

    @property
    def task_success(self) -> bool:
        return self.error is None and bool(self.output)

    @property
    def llm_cost(self) -> float:
        return self.input_tokens / 1e6 * INPUT_PRICE_PER_1M + self.output_tokens / 1e6 * OUTPUT_PRICE_PER_1M

    @property
    def tool_cost(self) -> float:
        return TOOL_COST_PER_CALL * len(self.tool_call_logs)

    @property
    def total_cost(self) -> float:
        return self.llm_cost + self.tool_cost


# ---------------------------------------------------------------------------
# 5. Run one scenario and read everything off the agent's message history
# ---------------------------------------------------------------------------
def is_correct_tool(tool_name: str, expected_tools: List[str], alternatives: List[List[str]]) -> bool:
    return tool_name in expected_tools or any(tool_name in alt for alt in alternatives)


def run_evaluation(scenario: dict) -> TaskEvaluation:
    expected = scenario["expected_tools"]
    alternatives = scenario.get("acceptable_alternatives", [])
    actual_tools, tool_logs = [], []
    output, error = None, None
    input_tokens = output_tokens = 0
    t0 = time.time()
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": scenario["question"]}]})
        messages = result["messages"]
        requested = {}                                           # tool_call_id -> the model's request
        for m in messages:
            if isinstance(m, AIMessage):
                for tc in m.tool_calls:
                    requested[tc["id"]] = tc
                if m.usage_metadata:                             # real token counts from OpenAI
                    input_tokens += m.usage_metadata["input_tokens"]
                    output_tokens += m.usage_metadata["output_tokens"]
            elif isinstance(m, ToolMessage):                     # one per tool that actually ran
                tc = requested.get(m.tool_call_id, {"name": m.name, "args": {}})
                actual_tools.append(tc["name"])
                text = str(m.content)
                tool_logs.append(ToolCallLog(
                    tool_name=tc["name"], parameters=tc["args"], output=text,
                    success=m.status != "error" and "error" not in text.lower(),
                    is_correct=is_correct_tool(tc["name"], expected, alternatives),
                ))
        output = messages[-1].text
    except Exception as e:
        error = str(e)

    return TaskEvaluation(
        question=scenario["question"], category=scenario["category"],
        expected_tools=expected, actual_tools=actual_tools, tool_call_logs=tool_logs,
        output=output, error=error,
        requires_external_data=scenario["requires_external_data"],
        optimal_tool_count=scenario["optimal_tool_count"], acceptable_alternatives=alternatives,
        input_tokens=input_tokens, output_tokens=output_tokens, latency_s=time.time() - t0,
    )


# ---------------------------------------------------------------------------
# 6. Metrics
# ---------------------------------------------------------------------------
def compute_precision(results: List[TaskEvaluation]) -> Dict:
    """Precision = correct tool calls / all tool calls made."""
    total_calls = sum(len(r.actual_tools) for r in results)
    correct_calls = sum(log.is_correct for r in results for log in r.tool_call_logs)
    per_task = []
    for r in results:
        if r.actual_tools:
            per_task.append(sum(log.is_correct for log in r.tool_call_logs) / len(r.actual_tools))
        else:
            per_task.append(1.0 if not r.expected_tools else 0.0)
    return {"overall_precision": correct_calls / total_calls if total_calls else 1.0,
            "avg_task_precision": sum(per_task) / len(per_task)}


def compute_recall(results: List[TaskEvaluation]) -> Dict:
    """Recall = expected tools that were called / all expected tools."""
    total_expected = sum(len(r.expected_tools) for r in results)
    per_task, recalled_total = [], 0
    for r in results:
        if r.expected_tools:
            recalled = min(sum(1 for t in r.actual_tools if is_correct_tool(t, r.expected_tools, r.acceptable_alternatives)),
                           len(r.expected_tools))
            recalled_total += recalled
            per_task.append(recalled / len(r.expected_tools))
        else:
            per_task.append(1.0 if not r.actual_tools else 0.0)
    return {"overall_recall": recalled_total / total_expected if total_expected else 1.0,
            "avg_task_recall": sum(per_task) / len(per_task)}


def compute_avg_tool_calls(results: List[TaskEvaluation]) -> Dict:
    """How many calls per task, and how often that was too few, just right, or too many."""
    counts = {"too_few": 0, "optimal": 0, "too_many": 0}
    for r in results:
        n = len(r.actual_tools)
        key = "too_few" if n < r.optimal_tool_count else "optimal" if n == r.optimal_tool_count else "too_many"
        counts[key] += 1
    total = len(results)
    return {"avg_calls_per_task": sum(len(r.actual_tools) for r in results) / total,
            "efficiency_rate": {k: v / total for k, v in counts.items()}}


def compute_tool_success_rate(results: List[TaskEvaluation]) -> Dict:
    """Share of tool calls that ran without an error."""
    logs = [log for r in results for log in r.tool_call_logs]
    successful = sum(log.success for log in logs)
    return {"overall_success_rate": successful / len(logs) if logs else 1.0, "failed_calls": len(logs) - successful}


def numbers_in(text: str) -> List[float]:
    return [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))]


def compute_attribution(results: List[TaskEvaluation]) -> Dict:
    """Did the final answer actually USE a tool's result? Check the tool's main number appears in it."""
    tool_required = [r for r in results if r.requires_external_data]
    attributed = 0
    for r in tool_required:
        answer_numbers = numbers_in(r.output or "")
        for log in r.tool_call_logs:
            tool_numbers = numbers_in(log.output)
            if tool_numbers and any(abs(tool_numbers[0] - a) < 0.05 for a in answer_numbers):
                attributed += 1
                break
    n = len(tool_required)
    return {"attribution_rate": attributed / n if n else 0.0}


def compute_cost(results: List[TaskEvaluation]) -> Dict:
    total_cost = sum(r.total_cost for r in results)
    successful = [r for r in results if r.task_success]
    cost_per_success = total_cost / len(successful) if successful else 0.0
    classification = "Low" if cost_per_success < 0.001 else "Moderate" if cost_per_success < 0.01 else "High"
    return {"total_cost": total_cost, "cost_per_success": cost_per_success, "cost_classification": classification}


# ---------------------------------------------------------------------------
# 7. Run everything and print the report
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Tools: {[t.name for t in tools]}")
    print(f"Running {len(test_scenarios)} scenarios...\n")

    evaluation_results: List[TaskEvaluation] = []
    for i, scenario in enumerate(test_scenarios, 1):
        r = run_evaluation(scenario)
        evaluation_results.append(r)
        print(f"[{i:>2}/{len(test_scenarios)}] [{r.category}] {r.question}")
        for log in r.tool_call_logs:
            args = ", ".join(f"{k}={v}" for k, v in log.parameters.items())
            print(f"        → {log.tool_name}({args})\n          = {log.output}")
        if not r.tool_call_logs:
            print("        → (no tool call)")
        answer = (r.output or "").replace("\n", " ")
        print(f"        answer: {answer[:110]}{'...' if len(answer) > 110 else ''}")
        match = "✅" if sorted(r.actual_tools) == sorted(r.expected_tools) else "❌"
        status = "" if r.task_success else f"   ERROR: {r.error}"
        print(f"        expected {r.expected_tools or 'no tool'} {match}  {r.latency_s:.1f}s{status}\n")

    precision = compute_precision(evaluation_results)
    recall = compute_recall(evaluation_results)
    calls = compute_avg_tool_calls(evaluation_results)
    success = compute_tool_success_rate(evaluation_results)
    attribution = compute_attribution(evaluation_results)
    cost = compute_cost(evaluation_results)
    avg_latency = sum(r.latency_s for r in evaluation_results) / len(evaluation_results)

    summary = pd.DataFrame([
        {"Category": "Tool Invocation", "Metric": "Precision (Overall)",  "Value": f"{precision['overall_precision']:.3f}", "Description": "Correct calls / Total calls"},
        {"Category": "Tool Invocation", "Metric": "Precision (Avg/Task)", "Value": f"{precision['avg_task_precision']:.3f}", "Description": "Average precision across tasks"},
        {"Category": "Tool Invocation", "Metric": "Recall (Overall)",     "Value": f"{recall['overall_recall']:.3f}", "Description": "Expected tools that were called"},
        {"Category": "Tool Invocation", "Metric": "Recall (Avg/Task)",    "Value": f"{recall['avg_task_recall']:.3f}", "Description": "Average recall across tasks"},
        {"Category": "Efficiency",      "Metric": "Avg Calls per Task",   "Value": f"{calls['avg_calls_per_task']:.2f}", "Description": "Mean tool invocations"},
        {"Category": "Efficiency",      "Metric": "Optimal Call Rate",    "Value": f"{calls['efficiency_rate']['optimal']:.1%}", "Description": "Tasks with optimal tool count"},
        {"Category": "Efficiency",      "Metric": "Over-calling Rate",    "Value": f"{calls['efficiency_rate']['too_many']:.1%}", "Description": "Tasks with excess tool calls"},
        {"Category": "Efficiency",      "Metric": "Under-calling Rate",   "Value": f"{calls['efficiency_rate']['too_few']:.1%}", "Description": "Tasks with too few calls"},
        {"Category": "Success",         "Metric": "Tool Success Rate",    "Value": f"{success['overall_success_rate']:.3f}", "Description": "Calls that ran without error"},
        {"Category": "Success",         "Metric": "Failed Tool Calls",    "Value": str(success['failed_calls']), "Description": "Number of failed executions"},
        {"Category": "Success",         "Metric": "Attribution Rate",     "Value": f"{attribution['attribution_rate']:.3f}", "Description": "Answers that used a tool's result"},
        {"Category": "System",          "Metric": "Avg Latency",          "Value": f"{avg_latency:.1f}s", "Description": "Time per task, end to end"},
        {"Category": "Cost",            "Metric": "Total Cost",           "Value": f"${cost['total_cost']:.6f}", "Description": "LLM tokens + tool calls"},
        {"Category": "Cost",            "Metric": "Cost per Success",     "Value": f"${cost['cost_per_success']:.6f}", "Description": "Cost per successful task"},
        {"Category": "Cost",            "Metric": "Cost Classification",  "Value": cost['cost_classification'], "Description": "Low / Moderate / High"},
    ])
    print("=" * 80)
    print("CONSOLIDATED EVALUATION SUMMARY")
    print("=" * 80)
    print(summary.to_string(index=False))

    def short(names):                                # "reorder_point_calculator" -> "reorder_point"
        return ", ".join(n.replace("_calculator", "").replace("inventory_", "") for n in names) or "None"

    rows = []
    for i, r in enumerate(evaluation_results, 1):
        exp_set, act_set = set(r.expected_tools), set(r.actual_tools)
        p = len(exp_set & act_set) / len(act_set) if act_set else (1.0 if not exp_set else 0.0)
        rc = len(exp_set & act_set) / len(exp_set) if exp_set else (1.0 if not act_set else 0.0)
        rows.append({"#": i, "Question": r.question[:42] + "...", "Category": r.category,
                     "Expected": short(r.expected_tools), "Actual": short(r.actual_tools),
                     "Precision": f"{p:.2f}", "Recall": f"{rc:.2f}", "Success": "Yes" if r.task_success else "No",
                     "Tokens": r.input_tokens + r.output_tokens, "Cost": f"${r.total_cost:.5f}"})
    print("\n" + "=" * 100)
    print("PER-TASK DETAILED RESULTS")
    print("=" * 100)
    print(pd.DataFrame(rows).to_string(index=False))
