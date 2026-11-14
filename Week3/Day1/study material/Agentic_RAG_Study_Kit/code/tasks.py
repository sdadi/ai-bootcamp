"""
tasks.py  -  The "to-do list" for each agent
============================================
Created by Shivank Agrawal

An AGENT is WHO does the work. A TASK is WHAT work must be done.

Every Task has:
  description      -> the instructions (a prompt). {question} is a placeholder
                      that CrewAI fills in with the real user question.
  expected_output  -> what the result must look like. Being strict here
                      ("answer only yes or no") makes the next agent's job easy.
  agent            -> who does this task
  context          -> outputs of EARLIER tasks this task is allowed to read.
                      This is how information flows from one agent to the next.
  tools            -> which tools may be used while doing this task

The chain looks like this:

  router_task -> retriever_task -> grader_task -> hallucination_task -> answer_task
   (where?)       (fetch it)       (relevant?)     (grounded?)            (final reply)
"""

from crewai import Agent, Task


def create_router_task(router_agent: Agent) -> Task:
    return Task(
        description=(
            "Analyse this question: '{question}'.\n"
            "Decide where its answer is most likely to be found:\n"
            "- Answer 'vectorstore' if the question is about a topic covered by our "
            "document (even if the exact words are different).\n"
            "- Answer 'websearch' if the question needs current or real-time "
            "information (news, prices, scores, today's events) or is about a general "
            "topic that our document does not cover.\n"
            "Decide using the semantic meaning of the question, not keyword matching."
        ),
        expected_output=(
            "Exactly one word: 'vectorstore' or 'websearch'. "
            "No explanation, no punctuation, nothing else."
        ),
        agent=router_agent,
    )


def create_retriever_task(retriever_agent: Agent, router_task: Task,
                          rag_tool, web_search_tool) -> Task:
    return Task(
        description=(
            "Read the router's decision from the context.\n"
            "- If it is 'vectorstore', call the rag_tool and pass it the question: "
            "'{question}'.\n"
            "- If it is 'websearch', call the web_search_tool and pass it the "
            "question: '{question}'.\n"
            "- If the rag_tool returns NOT_FOUND_IN_DOCUMENT, fall back to the "
            "web_search_tool with the same question.\n"
            "Always pass the user's full question to the tool."
        ),
        expected_output=(
            "First line: 'SOURCE: document' or 'SOURCE: web'.\n"
            "Then the answer to '{question}' based only on what the tool returned.\n"
            "Then the supporting text (document chunks or web snippets) copied as is."
        ),
        agent=retriever_agent,
        context=[router_task],
        tools=[rag_tool, web_search_tool],
    )


def create_grader_task(grader_agent: Agent, retriever_task: Task) -> Task:
    return Task(
        description=(
            "Look at the retrieved answer in the context and the question "
            "'{question}'. Decide whether the retrieved answer is RELEVANT to the "
            "question, i.e. it actually addresses what was asked."
        ),
        expected_output=(
            "Exactly one word: 'yes' if the retrieved answer is relevant to the "
            "question, or 'no' if it is not. No explanation."
        ),
        agent=grader_agent,
        context=[retriever_task],
    )


def create_hallucination_task(hallucination_agent: Agent, retriever_task: Task,
                              grader_task: Task) -> Task:
    return Task(
        description=(
            "Using the context, compare the retrieved answer with the supporting "
            "text (document chunks or web snippets) it came from. Decide whether "
            "the answer is grounded in / supported by that text for the question "
            "'{question}'. Also note the relevance grade from the previous check."
        ),
        expected_output=(
            "Exactly one word: 'yes' if the answer is supported by the source text "
            "and is useful (NO hallucination), or 'no' if it contains claims that the "
            "source text does not support (hallucination). No explanation."
        ),
        agent=hallucination_agent,
        context=[retriever_task, grader_task],
    )


def create_answer_task(answer_agent: Agent, retriever_task: Task, grader_task: Task,
                       hallucination_task: Task, web_search_tool) -> Task:
    return Task(
        description=(
            "Write the final reply to the question: '{question}'.\n"
            "Read the context: the retrieved answer, the relevance grade and the "
            "hallucination grade.\n"
            "- If BOTH grades are 'yes', return a clear and concise answer based on "
            "the retrieved answer.\n"
            "- If EITHER grade is 'no', call the web_search_tool with the question "
            "'{question}' and answer from those results instead.\n"
            "Make sure the final answer contains only the answer to the question, "
            "not an explanation of how you found it."
        ),
        expected_output=(
            "A clear and concise answer to '{question}' in plain language "
            "(about 3-6 sentences)."
        ),
        agent=answer_agent,
        context=[retriever_task, grader_task, hallucination_task],
        tools=[web_search_tool],
    )
