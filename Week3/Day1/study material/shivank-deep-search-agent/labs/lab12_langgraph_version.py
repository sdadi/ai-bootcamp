"""
LAB 12 — The same deep-search loop as a LangGraph state machine (optional)
================================================================================
Course module: 2 + 12 (Architecture, frameworks)

Our agent is plain Python so you can read every line. Frameworks such as LangGraph express the
same idea as a GRAPH: nodes do work, edges decide what runs next, and a shared STATE object is
passed along. The "continue or stop" decision becomes a conditional edge.

    START → plan → execute → validate_merge → evaluate ─┬─ continue → execute
                                                        └─ stop     → synthesise → END

Why bother? Graph frameworks give you checkpoints (resume after a crash), human-in-the-loop
pauses, streaming of intermediate steps and visual debugging — useful once agents run for
minutes and cost real money.

Run it
  pip install langgraph        # already in requirements.txt
  python labs/lab12_langgraph_version.py
"""

from _common import banner, pause_note, step, table  # noqa: E402

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from deepsearch.budget import Budget
from deepsearch.config import SETTINGS, describe
from deepsearch.executor import Executor
from deepsearch.llm import LLM
from deepsearch.merger import merge_all
from deepsearch.orchestrator import DEFAULT_QUESTION, DeepSearchAgent, default_entity
from deepsearch.planner import create_plan, evaluate_progress, review_plan


class ResearchState(TypedDict, total=False):
    question: str
    plan: Any
    open_ids: list[str]
    iteration: int
    new_evidence: int
    merged: dict
    conflicts: list
    decision: Any
    report: Any
    log: list[str]


def build_graph(agent: DeepSearchAgent):
    """Each node is a thin wrapper around the SAME functions the plain-Python agent uses."""
    budget = agent.budget
    executor = Executor(agent.entity, budget, max_parallel=SETTINGS.max_parallel,
                        results_per_query=SETTINGS.results_per_query)
    agent.executor = executor

    def plan_node(state: ResearchState) -> dict:
        plan, review = review_plan(create_plan(LLM(budget=budget), state["question"], agent.entity))
        plan.entity = agent.entity
        agent.plan = plan
        return {"plan": plan, "open_ids": [q.id for q in plan.sub_questions], "iteration": 0,
                "log": [f"plan: {len(plan.sub_questions)} sub-questions"]}

    def execute_node(state: ResearchState) -> dict:
        it = state["iteration"] + 1
        todo = [q for q in state["plan"].sub_questions if q.id in state["open_ids"]]
        before = len(agent.evidence)
        agent.evidence.extend(executor.run(todo, it))
        return {"iteration": it, "new_evidence": len(agent.evidence) - before,
                "log": state["log"] + [f"execute #{it}: {len(todo)} workers"]}

    def validate_merge_node(state: ResearchState) -> dict:
        agent.validate(executor)
        merged, conflicts = merge_all(agent.evidence)
        return {"merged": merged, "conflicts": conflicts,
                "log": state["log"] + [f"validate+merge: {sum(e.accepted for e in agent.evidence)} accepted"]}

    def evaluate_node(state: ResearchState) -> dict:
        unresolved = {c.field for c in state["conflicts"] if c.resolution == "unresolved"}
        d = evaluate_progress(state["plan"], state["merged"], unresolved, state["iteration"], agent.max_iterations,
                              state["new_evidence"], budget)
        return {"decision": d, "open_ids": d.open_sub_questions,
                "log": state["log"] + [f"evaluate: {d.action} ({d.reasons[0] if d.reasons else ''})"]}

    def synthesise_node(state: ResearchState) -> dict:
        report = agent.synthesise(state["merged"], state["conflicts"], executor, budget.elapsed)
        return {"report": report, "log": state["log"] + ["synthesise: report ready"]}

    def route(state: ResearchState) -> str:  # the conditional edge = the planner's stop rule
        return "continue" if state["decision"].action == "continue" else "stop"

    g = StateGraph(ResearchState)
    g.add_node("plan", plan_node)
    g.add_node("execute", execute_node)
    g.add_node("validate_merge", validate_merge_node)
    g.add_node("evaluate", evaluate_node)
    g.add_node("synthesise", synthesise_node)
    g.add_edge(START, "plan")
    g.add_edge("plan", "execute")
    g.add_edge("execute", "validate_merge")
    g.add_edge("validate_merge", "evaluate")
    g.add_conditional_edges("evaluate", route, {"continue": "execute", "stop": "synthesise"})
    g.add_edge("synthesise", END)
    return g.compile()


def main() -> None:
    banner("LAB 12 — Deep search as a LangGraph state machine")
    print("  Settings:", describe())
    SETTINGS.results_per_query = 1  # tight reading budget so you can see the loop go round twice
    agent = DeepSearchAgent(DEFAULT_QUESTION, default_entity(), budget=Budget(), use_memory=False)
    graph = build_graph(agent)

    step("Graph structure (Mermaid — paste into https://mermaid.live to see it)")
    try:
        print(graph.get_graph().draw_mermaid())
    except Exception as exc:  # drawing is optional
        print("  (could not draw:", exc, ")")

    step("Run")
    final = graph.invoke({"question": DEFAULT_QUESTION})
    for line in final["log"]:
        print("  •", line)
    report = final["report"]
    table([{"field": f.field, "value": f.value, "confidence": f.confidence} for f in report.key_findings])
    pause_note("Same components, different control-flow engine. Learn the plain-Python version first; reach for a "
               "framework when you need checkpoints, human approval steps or streaming.")


if __name__ == "__main__":
    main()
