"""
agents.py  -  Our team of 5 AI "employees"
==========================================
Created by Shivank Agrawal

Every agent in CrewAI is described by 3 things, like a job description:

  role       -> the job title             ("Router")
  goal       -> what success looks like   ("send the question to the right place")
  backstory  -> personality + expertise   ("You are an expert in ...")

The backstory is pure PROMPT ENGINEERING. If an agent behaves badly,
this is the first place you tweak. Small wording changes = big behaviour changes.

  llm               -> the brain this agent thinks with
  allow_delegation  -> may this agent hand its work to another agent?
                       We say False: each agent does exactly its own job.
  max_iter          -> how many think-act rounds the agent may take before it
                       must answer. A safety brake against endless loops.
  verbose           -> True = print the agent's thinking in the terminal
                       (great for learning and debugging)

The team (in the order they work), with their real-life twins:
  1. Router               - hospital reception: sends you to the right department
  2. Retriever            - librarian: fetches the right book / searches online
  3. Answer Grader        - teacher: "did you answer the question that was asked?"
  4. Hallucination Grader - auditor: "show me the receipt for every claim"
  5. Answer Generator     - news anchor: presents only the checked final story
"""

from crewai import Agent

import config
from llms import get_main_llm


def create_router_agent() -> Agent:
    return Agent(
        role="Router",
        goal="Route each user question to either the vectorstore (our PDF) "
        "or a web search, based on what the question is really about.",
        backstory=(
            "You are an expert at deciding whether a question can be answered "
            "from the information stored in our vector database or needs a web search. "
            f"The vector database contains a detailed document about {config.PDF_TOPIC}. "
            "You make routing decisions based on the semantic MEANING of the "
            "question, not just on matching keywords."
        ),
        allow_delegation=False,
        max_iter=5,
        verbose=True,
        llm=get_main_llm(),  # the DeepSeek brain (same one for all agents)
    )


def create_retriever_agent() -> Agent:
    return Agent(
        role="Retriever",
        goal="Use the tool chosen by the router to fetch the information "
        "needed to answer the question.",
        backstory=(
            "You are an assistant for question-answering tasks. You call exactly "
            "the tool you are told to call, pass it the user's full question, and "
            "report what the tool returned. Use the information present in the "
            "retrieved context to answer the question. Provide a clear, concise answer "
            "and never add facts that are not in the retrieved context."
        ),
        allow_delegation=False,
        max_iter=5,
        verbose=True,
        llm=get_main_llm(),  # the DeepSeek brain (same one for all agents)
    )


def create_answer_grader_agent() -> Agent:
    return Agent(
        role="Answer Grader",
        goal="Filter out wrong or irrelevant retrievals.",
        backstory=(
            "You are a strict grader. You read the question and the retrieved answer "
            "and judge only one thing: does this answer actually address the question? "
            "If it contains the key ideas related to the question, it is relevant."
        ),
        allow_delegation=False,
        max_iter=5,
        verbose=True,
        llm=get_main_llm(),  # the DeepSeek brain (same one for all agents)
    )


def create_hallucination_grader_agent() -> Agent:
    return Agent(
        role="Hallucination Grader",
        goal="Catch made-up facts before they reach the user.",
        backstory=(
            "You are a fact checker. You compare the answer with the source text it "
            "came from and decide whether every important claim is supported by that "
            "source. An answer that invents facts, numbers or names is a hallucination."
        ),
        allow_delegation=False,
        max_iter=5,
        verbose=True,
        llm=get_main_llm(),  # the DeepSeek brain (same one for all agents)
    )


def create_answer_generator_agent() -> Agent:
    return Agent(
        role="Answer Generator",
        goal="Give the user one clear, correct and concise final answer.",
        backstory=(
            "You are the final editor. You receive the retrieved answer together with "
            "the relevance and hallucination checks. If the checks pass, you polish the "
            "answer. If they fail, you run a fresh web search and answer from that. "
            "You never explain your internal process to the user."
        ),
        allow_delegation=False,
        max_iter=5,
        verbose=True,
        llm=get_main_llm(),  # the DeepSeek brain (same one for all agents)
    )
