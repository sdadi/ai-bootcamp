"""
Exercise 3.2 — The same Tool Calling, with LangChain
Created by Shivank Agarwal

Same 4 steps as Exercise 3.1, but LangChain removes the boring parts:
  * @tool builds the JSON "menu" for us from the function name, type hints and docstring
  * llm.bind_tools([...]) attaches the menu to every call
  * response.tool_calls gives the requests already parsed into Python dicts

Run it:
    python 2_tool_calling_with_langchain.py
"""

import math

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()


# ----- TODO 1: Turn square_root into a LangChain tool -----
# Hint: Put the @tool decorator above the function
# Hint: The docstring becomes the tool description the LLM reads, so keep it clear
def square_root(number: float) -> float:
    """Calculate the square root of a non-negative number."""
    return math.sqrt(number)  # add the decorator above this function


tools = [square_root]
tool_map = {t.name: t for t in tools}  # "square_root" -> the tool object

llm = ChatOpenAI(model="gpt-6-luna", reasoning_effort="none")
llm_with_tools = llm.bind_tools(tools)  # the LLM now knows the tool exists


def chat_with_tools(user_message: str) -> str:
    messages = [HumanMessage(content=user_message)]

    response = llm_with_tools.invoke(messages)  # Step 1: ask
    messages.append(response)

    if not response.tool_calls:  # Step 2: no tool needed
        # .text = the answer as plain text. (With tools bound, LangChain talks to OpenAI's newer
        # "Responses API", where .content is a list of blocks; .text joins them for us.)
        return response.text

    # ----- TODO 2: Run every requested tool and add a ToolMessage with its result -----
    # Hint: Each tool call is a dict with keys "name", "args" and "id"
    # Hint: tool_map[tc["name"]].invoke(tc["args"]) runs the tool
    # Hint: messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    pass  # replace with your code

    return llm_with_tools.invoke(messages).text  # Step 4: final answer (.text = plain text)


if __name__ == "__main__":
    # Peek at the JSON "menu" that @tool generated for us
    print("Tool name       :", square_root.name)
    print("Tool description:", square_root.description)
    print("Tool arguments  :", square_root.args)

    questions = [
        "Is 15241383937 the square of 123456?",  # a trap: 123456 squared is 15241383936
        "What is the capital of France?",
        "Can you calculate the square root of 2025 for me?",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {chat_with_tools(q)}")
