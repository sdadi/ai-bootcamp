"""
Agentic Bug Fixing  -  OpenAI version
=====================================
Author: Shivank Agarwal

WHAT THIS PROGRAM DOES (one line)
    It runs a Python function. If the function crashes, OpenAI GPT rewrites the
    function so it handles the error, then the new version is run again.

HOW IT WORKS (the "agent loop")

        START
          |
          v
    +----------------------+   no error / too many tries    +-----+
    | code_execution_node  | -----------------------------> | END |
    |  (run the function)  |                                +-----+
    +----------------------+
          |  it crashed
          v
    +----------------------+
    |    code_fix_node     |   asks GPT to rewrite the function,
    |  (ask GPT to fix it) |   then loads the new code
    +----------------------+
          |
          +------> back to code_execution_node (run the NEW version)

HOW TO RUN
    1.  pip install -r requirements.txt
    2.  Create a file called .env next to this file with one line:
            OPENAI_API_KEY=sk-...your key...
    3.  python agentic_bug_fixing_openai.py          -> quick demo in the terminal
        python agentic_bug_fixing_openai.py --ui     -> browser app (Gradio) with a diff viewer

SAFETY NOTE
    This program runs (exec) code that GPT wrote, on your own machine.
    That is fine for a classroom demo. In a real product, run it inside a sandbox
    (a Docker container, E2B, a locked-down VM...) so bad code can't hurt anything.
"""

# ---------------------------------------------------------------------------
# STEP 0: Imports and settings
# ---------------------------------------------------------------------------
import ast          # reads Python code as a tree, so we can find the "def" name
import difflib      # makes a git-style "before vs after" diff
import html         # escapes text so it is safe to show inside HTML
import inspect      # gets the source code (text) of a Python function
import json         # parses the arguments the user types in the UI, e.g. [10, 0]
import re           # regex, used to strip ```python fences from GPT's reply
import sys          # reads the --ui command-line flag
import textwrap     # removes extra indentation from code
from typing import Any, Callable

from dotenv import load_dotenv                         # loads OPENAI_API_KEY from the .env file
from langchain_core.prompts import ChatPromptTemplate  # a prompt with {blanks} to fill
from langchain_openai import ChatOpenAI                # <-- the OpenAI chat model (was ChatGroq)
from langgraph.graph import StateGraph, END            # builds the agent's flowchart
from pydantic import BaseModel                         # defines the shape of our "State"

load_dotenv(override=True)

MODEL_NAME = "gpt-5-mini"   # the OpenAI model that writes the fixes
MAX_FIX_ATTEMPTS = 3        # safety brake: never ask GPT more than 3 times for one function


# ---------------------------------------------------------------------------
# STEP 1: The State  =  the "job card" every node reads from and writes into
# ---------------------------------------------------------------------------
class State(BaseModel):
    function: Callable              # the actual Python function object we will run
    function_string: str            # the ORIGINAL source code, as text
    arguments: list                 # inputs to call the function with, e.g. [10, 0]
    error: bool = False             # did the last run crash?
    error_description: str = ""     # the crash message, e.g. "ZeroDivisionError: division by zero"
    new_function_string: str = ""   # the fixed code written by GPT ("" = no fix needed)
    attempts: int = 0               # how many times we have asked GPT to fix it
    result: Any = None              # what the function returned on its last good run


# ---------------------------------------------------------------------------
# STEP 2: The prompt we send to GPT
# ---------------------------------------------------------------------------
# Words inside {curly braces} are blanks that get filled in at run time.
FIX_PROMPT = ChatPromptTemplate.from_template(
    "You are fixing a Python function that crashed.\n\n"
    "Function:\n{function_string}\n\n"
    "It was called with these arguments: {arguments}\n"
    "It raised this error: {error_description}\n\n"
    "Rewrite the function so it handles this error gracefully by returning an "
    "error message (a string) instead of crashing.\n"
    "Rules:\n"
    "- Use the exact same function name and parameters.\n"
    "- Return only the function definition: no explanation, no markdown, no ``` fences."
)


# ---------------------------------------------------------------------------
# STEP 3: Small helper functions
# ---------------------------------------------------------------------------
def clean_llm_code(text: str) -> str:
    """GPT sometimes wraps code in ```python ... ``` even when told not to. Remove that."""
    text = text.strip()
    fenced = re.search(r"```(?:python|py)?[ \t]*\n(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    # dedent: if the code arrives indented, exec() would fail with "unexpected indent"
    return textwrap.dedent(text).strip()


def first_function_name(code: str) -> str:
    """Return the name of the first `def` in a piece of code, e.g. 'divide_two_numbers'."""
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
    raise ValueError("No function definition (def ...) found in the code.")


def load_function_from_code(code: str, function_name: str | None = None) -> Callable:
    """Turn code TEXT into a real function OBJECT we can call.

    exec() runs the text as Python. The `def` inside it creates the function and
    stores it in the `namespace` dictionary, from where we pick it up by name.
    """
    namespace: dict = {}
    exec(code, namespace)  # <-- runs GPT-written code: see the SAFETY NOTE at the top
    name = function_name or first_function_name(code)
    function = namespace.get(name)
    if not callable(function):
        raise ValueError(f"The code does not define a function called '{name}'.")
    return function


# ---------------------------------------------------------------------------
# STEP 4: The Agent  =  two nodes + one router, wired into a LangGraph graph
# ---------------------------------------------------------------------------
class Agent:
    def __init__(self, model_name: str = MODEL_NAME):
        self.llm = ChatOpenAI(model=model_name)   # <-- the swap: was ChatGroq(model=...)
        self.fix_chain = FIX_PROMPT | self.llm     # "pipe" = fill the prompt, then send it to GPT
        self.graph = self.build_graph()

    # ---- Node 1: run the function and record what happened -----------------
    def code_execution_node(self, state: State) -> dict:
        print(f"[run]  {state.function.__name__}{tuple(state.arguments)}")
        try:
            result = state.function(*state.arguments)
        except Exception as e:
            # Keep the error TYPE too: "KeyError: 'INVALID'" tells GPT far more than "'INVALID'"
            description = f"{type(e).__name__}: {e}"
            print(f"[run]  crashed -> {description}")
            return {"error": True, "error_description": description}
        print(f"[run]  ok -> {result!r}")
        return {"error": False, "error_description": "", "result": result}
        # A node returns ONLY the fields it changed. LangGraph merges them into the State.

    # ---- Node 2: ask GPT to fix the function --------------------------------
    def code_fix_node(self, state: State) -> dict:
        attempt = state.attempts + 1
        print(f"[fix]  attempt {attempt}/{MAX_FIX_ATTEMPTS}: asking {self.llm.model_name} to fix it")

        # If an earlier fix still crashed, send the LATEST code, not the original one.
        code_to_fix = state.new_function_string or state.function_string

        reply = self.fix_chain.invoke({
            "function_string": code_to_fix,
            "arguments": json.dumps(state.arguments, default=str),
            "error_description": state.error_description,
        })
        new_code = clean_llm_code(reply.content)

        try:
            new_function = load_function_from_code(new_code, state.function.__name__)
        except Exception as e:
            # GPT wrote code that doesn't even load. Count the attempt and keep the
            # previous version; the loop will run it, see the same crash, and try again.
            print(f"[fix]  the new code could not be loaded ({type(e).__name__}: {e})")
            return {"attempts": attempt}

        print("[fix]  new version loaded, running it again")
        return {"attempts": attempt, "function": new_function, "new_function_string": new_code}

    # ---- Router: after each run, decide where to go next ---------------------
    def error_router(self, state: State) -> str:
        if not state.error:
            return END                                  # it worked, we're done
        if state.attempts >= MAX_FIX_ATTEMPTS:
            print(f"[stop] still failing after {state.attempts} fix attempts, giving up")
            return END                                  # safety brake
        return "code_fix_node"                          # crashed: go fix it

    # ---- Wire the nodes together ---------------------------------------------
    def build_graph(self):
        builder = StateGraph(State)
        builder.add_node("code_execution_node", self.code_execution_node)
        builder.add_node("code_fix_node", self.code_fix_node)

        builder.set_entry_point("code_execution_node")                 # always start by running it
        builder.add_conditional_edges("code_execution_node", self.error_router)  # then ask the router
        builder.add_edge("code_fix_node", "code_execution_node")       # after a fix, run again

        return builder.compile()

    # ---- Easy entry points ----------------------------------------------------
    def run(self, function: Callable, arguments: list, function_string: str | None = None) -> dict:
        """Run the whole loop. Returns the final State as a dictionary."""
        state = State(
            function=function,
            function_string=function_string or textwrap.dedent(inspect.getsource(function)),
            arguments=arguments,
        )
        return self.graph.invoke(state)

    def execute(self, function: Callable, arguments: list):
        """Same as the original execute(): give back the function's answer."""
        final_state = self.run(function, arguments)
        if final_state.get("error"):
            return (f"Could not fix the function after {final_state.get('attempts', 0)} attempt(s). "
                    f"Last error: {final_state.get('error_description')}")
        return final_state.get("result")


# ---------------------------------------------------------------------------
# STEP 5: The browser app (Gradio) - type code + arguments, see the fix as a diff
# ---------------------------------------------------------------------------
class App:
    def __init__(self):
        self.agent = Agent()
        self.original_code = ""
        self.fixed_code = ""

    def run_agent_and_show_diff(self, code_string: str, arguments_str: str):
        """Button handler. Returns 4 things: original code, fixed code, diff HTML, result text."""

        # 5a. Read the arguments the user typed, e.g. [10, 0]
        try:
            arguments = json.loads(arguments_str)
            if not isinstance(arguments, list):
                arguments = [arguments]          # a single value becomes a 1-item list
        except json.JSONDecodeError:
            return code_string, "Error parsing arguments", self._error_html("Invalid JSON format for arguments"), "Error: Invalid arguments"

        # 5b. Turn the code text into a real function
        try:
            function = load_function_from_code(code_string)
        except Exception as e:
            return code_string, f"Error: {e}", self._error_html(f"Error loading code: {e}"), f"Error: {e}"

        # 5c. Run the agent loop
        self.original_code = code_string
        try:
            final_state = self.agent.run(function, arguments, function_string=code_string)
        except Exception as e:  # e.g. wrong API key, no internet
            return code_string, "Agent failed", self._error_html(f"Agent failed: {e}"), f"Error: {e}"

        # 5d. Build the outputs
        self.fixed_code = final_state.get("new_function_string", "")
        if self.fixed_code:
            diff = self._generate_colored_diff_html(self.original_code, self.fixed_code)
        else:
            diff = ('<pre style="font-family: monospace; background-color: #f6f8fa; padding: 10px; '
                    'border-radius: 5px; color: #1a7f37;">No errors detected - code executed successfully!</pre>')

        if final_state.get("error"):
            result = (f"Still failing after {final_state.get('attempts', 0)} fix attempt(s): "
                      f"{final_state.get('error_description')}")
        else:
            result = repr(final_state.get("result"))

        return self.original_code, self.fixed_code or "No fix needed", diff, result

    @staticmethod
    def _error_html(message: str) -> str:
        return f'<pre style="color: red;">{html.escape(message)}</pre>'

    def _generate_diff(self, original: str, fixed: str) -> str:
        """Unified diff, like `git diff`: '-' = removed line, '+' = added line."""
        diff = difflib.unified_diff(
            original.splitlines(), fixed.splitlines(),
            fromfile="original", tofile="fixed", lineterm="",
        )
        return "\n".join(diff)

    def _generate_colored_diff_html(self, original: str, fixed: str) -> str:
        """Same diff, coloured red/green like GitHub."""
        diff_text = self._generate_diff(original, fixed)
        if not diff_text:
            return ('<pre style="font-family: monospace; background-color: #f6f8fa; padding: 10px; '
                    'border-radius: 5px;">No differences found</pre>')

        html_lines = []
        for line in diff_text.split("\n"):
            escaped = html.escape(line)
            if line.startswith("---") or line.startswith("+++"):
                html_lines.append(f'<div style="font-weight: bold;">{escaped}</div>')                         # file headers
            elif line.startswith("@@"):
                html_lines.append(f'<div style="color: #0969da;">{escaped}</div>')                           # line numbers
            elif line.startswith("-"):
                html_lines.append(f'<div style="background-color: #ffebe9; color: #cf222e;">{escaped}</div>')  # removed
            elif line.startswith("+"):
                html_lines.append(f'<div style="background-color: #dafbe1; color: #1a7f37;">{escaped}</div>')  # added
            else:
                html_lines.append(f"<div>{escaped}</div>")                                                     # unchanged

        return ('<pre style="font-family: monospace; background-color: #f6f8fa; padding: 10px; '
                f'border-radius: 5px; overflow-x: auto;">{"".join(html_lines)}</pre>')

    def launch(self):
        import gradio as gr

        # Ready-made buggy functions to try (name -> code + arguments as JSON)
        examples = {
            "Division by Zero": {
                "code": "def divide_two_numbers(a, b):\n    return a / b",
                "args": "[10, 0]",
            },
            "E-commerce: 100% Discount Price": {
                "code": (
                    "def calculate_total_price(items, discount_percent):\n"
                    "    subtotal = sum(item['price'] * item['quantity'] for item in items)\n"
                    "    discount_amount = subtotal * (discount_percent / 100)\n"
                    "    margin = subtotal / (subtotal - discount_amount)\n"
                    "    final_price = subtotal - discount_amount\n"
                    "    return final_price"
                ),
                "args": '[[{"price": 10, "quantity": 2}], 100]',
            },
            "Invalid Coupon Code": {
                "code": (
                    "def apply_coupon_code(cart_total, coupon_code):\n"
                    "    coupons = {\n"
                    "        'SAVE10': 0.10,\n"
                    "        'SAVE20': 0.20,\n"
                    "        'BLACKFRIDAY': 0.50\n"
                    "    }\n"
                    "    discount = coupons[coupon_code]\n"
                    "    return cart_total * (1 - discount)"
                ),
                "args": '[100, "INVALID"]',
            },
            "Product Index Out of Range": {
                "code": "def get_product_price(products, product_id):\n    return products[product_id]['price']",
                "args": '[[{"price": 10}, {"price": 20}], 5]',
            },
            "Bulk Discount - Zero Quantity": {
                "code": (
                    "def calculate_bulk_discount(quantity, unit_price):\n"
                    "    avg_discount = 100 / quantity\n"
                    "    if quantity >= 100:\n"
                    "        discount = 0.20\n"
                    "    elif quantity >= 50:\n"
                    "        discount = 0.10\n"
                    "    elif quantity >= 10:\n"
                    "        discount = 0.05\n"
                    "    else:\n"
                    "        discount = 0.0\n"
                    "    return quantity * unit_price * (1 - discount)"
                ),
                "args": "[0, 10.99]",
            },
            "Format Price - None Value": {
                "code": 'def format_price_display(price):\n    return f"${price:.2f}"',
                "args": "[null]",
            },
            # This one does NOT crash: .get() quietly returns None. The agent only fixes
            # crashes, so it reports "no errors". Good discussion point about silent bugs!
            "Inventory - Missing SKU (silent bug)": {
                "code": "def get_inventory_count(inventory, sku):\n    return inventory.get(sku)",
                "args": '[{"ABC123": 10}, "XYZ999"]',
            },
        }
        first = "Division by Zero"

        def load_example(example_name):
            if example_name in examples:
                return examples[example_name]["code"], examples[example_name]["args"]
            return "", "[]"

        with gr.Blocks(title="Agentic Bug Fixing - Code Diff Viewer") as demo:
            gr.Markdown("# Agentic Bug Fixing - Code Diff Viewer")
            gr.Markdown("Enter a Python function and its arguments. If it crashes, the agent asks "
                        f"OpenAI (`{self.agent.llm.model_name}`) to fix it.")

            example_dropdown = gr.Dropdown(choices=list(examples.keys()), label="Load Example", value=first)

            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### Python Function Code")
                    code_input = gr.Code(value=examples[first]["code"], language="python", label="", lines=12)
                with gr.Column(scale=1):
                    gr.Markdown("### Function Arguments")
                    arguments_input = gr.Textbox(value=examples[first]["args"], label="Arguments (JSON format)",
                                                 placeholder='[10, 0] or {"key": "value"}', lines=2)
                    run_btn = gr.Button("Run Agent & Show Diff", variant="primary", size="lg")

            gr.Markdown("### Git-Style Diff")
            diff_output = gr.HTML(label="")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Original Code")
                    original_output = gr.Code(language="python", label="")
                with gr.Column():
                    gr.Markdown("### Fixed Code")
                    fixed_output = gr.Code(language="python", label="")

            gr.Markdown("### Execution Result")
            result_output = gr.Textbox(label="", lines=3)

            example_dropdown.change(fn=load_example, inputs=[example_dropdown],
                                    outputs=[code_input, arguments_input])
            run_btn.click(fn=self.run_agent_and_show_diff, inputs=[code_input, arguments_input],
                          outputs=[original_output, fixed_output, diff_output, result_output])

        demo.launch()


# ---------------------------------------------------------------------------
# STEP 6: Terminal demo (what runs when you type: python agentic_bug_fixing_openai.py)
# ---------------------------------------------------------------------------
def demo_in_terminal():
    agent = Agent()

    def divide_two_numbers(a, b):
        return a / b

    print("\n=== 1) The ORIGINAL function, called directly ===")
    try:
        divide_two_numbers(10, 0)
    except ZeroDivisionError as e:
        print(f"Crashed as expected: ZeroDivisionError: {e}")
    # (The original script did `answer_2 = divide_two_numbers(10, 0)` without try/except,
    #  so the whole program died with a traceback at the very end.)

    print("\n=== 2) The SAME call, through the bug-fixing agent ===")
    final_state = agent.run(divide_two_numbers, [10, 0])

    print("\n--- Fixed code written by GPT ---")
    print(final_state.get("new_function_string") or "(no fix was needed)")
    print("\n--- Final answer ---")
    if final_state.get("error"):
        print(f"Still failing: {final_state.get('error_description')}")
    else:
        print(repr(final_state.get("result")))


if __name__ == "__main__":
    if "--ui" in sys.argv:
        App().launch()
    else:
        demo_in_terminal()
