"""
Exercise 3.1 — Tool Calling from Scratch (plain OpenAI SDK, no frameworks)
Created by Shivank Agarwal

LLMs are good with words but can be wrong with exact maths.
"Tool calling" lets the LLM ask OUR Python code to do a job, then use the result.

The flow (remember these 4 steps):
  1. We send the question + a description of our tools (a JSON "menu")
  2. The LLM replies either with a normal answer OR with "please call tool X with arguments Y"
  3. Our code runs the Python function and sends the result back (role = "tool")
  4. The LLM writes the final answer using that result

Run it:
    python 1_tool_calling_from_scratch.py
"""

import json
import math

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # reads OPENAI_API_KEY from the .env file

client = OpenAI()
MODEL = "gpt-6-luna"

# ----------------------------------------------------------------------------
# 1) The real Python function (the "tool")
# ----------------------------------------------------------------------------
def square_root(number: float) -> float:
    if number < 0:
        raise ValueError("Cannot compute square root of a negative number.")
    return math.sqrt(number)


# Maps the tool NAME the LLM will use -> the Python function we run
tool_map = {"square_root": square_root}

# ----------------------------------------------------------------------------
# 2) The tool "menu" we show the LLM (JSON Schema)
#    The LLM never sees our Python code. It only sees this name, description and parameters.
# ----------------------------------------------------------------------------
# ----- TODO 1: Describe the square_root tool in OpenAI's JSON tool format -----
# Hint: A list with one dict: {"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}
# Hint: "parameters" is a JSON Schema: {"type": "object", "properties": {"number": {"type": "number", ...}}, "required": ["number"]}
tools = [{
    "type": "function",
    "function": {
        "name": "square_root",
        "description": "Calculates the square root of a non-negative number.",
        "parameters": {
            "type": "object",
            "properties": {
                "number": {
                    "type": "number",
                    "description": "The non-negative number to compute the square root of.",
                }
            },
            "required": ["number"],
        },
    },
}]


def chat_with_tools(user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]

    # Step 1: ask the LLM, and tell it which tools exist.
    # reasoning_effort="none" is required for tool calling with this model in the Chat Completions API.
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        reasoning_effort="none",
    )
    message = response.choices[0].message
    messages.append(message)  # keep the assistant's reply (including its tool requests) in the history

    # Step 2: did the LLM ask for a tool?
    if not message.tool_calls:
        return message.content  # no tool needed, e.g. "What is the capital of France?"

    # Step 3: run every tool the LLM asked for and send the results back
    for tool_call in message.tool_calls:
        # ----- TODO 2: Run the requested tool and add its result to messages -----
        # Hint: The tool name is in tool_call.function.name
        # Hint: The arguments are a JSON string in tool_call.function.arguments
        # Hint: json.loads(...) turns the arguments into a dict, then call tool_map[name](**args)
        # Hint: Append {"role": "tool", "tool_call_id": tool_call.id, "content": str(result)}
        name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        result = tool_map[name](**arguments)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(result),
        })

    # Step 4: ask again; now the LLM can see the tool results and write the final answer
    # ----- TODO 3: Call the LLM a second time with the updated messages and return its text -----
        final_response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        reasoning_effort="none",
    )
    
    # Hint: Same client.chat.completions.create(...) call as in Step 1, with the updated messages
    # return "TODO"  # replace with your code
    return final_response.choices[0].message.content


if __name__ == "__main__":
    questions = [
        "What is the square root of 144?",
        "What is the capital of France?",               # no tool needed
        "Can you calculate the square root of 2025 for me?",
        "What is the square root of -9?",                # negative number, should raise an error
    ]
    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {chat_with_tools(q)}")
