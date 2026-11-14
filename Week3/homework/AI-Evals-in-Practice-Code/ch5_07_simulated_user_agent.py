"""
Chapter 5.8 · Evaluating agents with simulated users (a mini τ²-bench)
AI Evals in Practice · Author: Shivank Agrawal

Agents act: they call tools that CHANGE things. So we test them the way τ²-bench does,
with three files in data/course_admin/:
  policy.md   the rules the agent must follow
  db.json     the "world" that the agent's tools read and change
  task.json   a simulated user (persona + hidden goal) and how to grade the episode

The agent (gpt-6.1-terra, with tools) and a simulated student (gpt-6-astra) chat until the
student stops. Then we grade the episode on three things:
  1. the END STATE of the database   (did the 3rd extension get granted? it must not)
  2. the ACTIONS taken                (was the student transferred to a human?)
  3. the CONVERSATION, with a judge   (was the rule explained without blaming the student?)
Reward = 1 only if every check passes.

Run:  python ch5_07_simulated_user_agent.py
"""
import json
from pathlib import Path

from openai import OpenAI

from judges import Verdict, judge

client = OpenAI()

DIR = Path(__file__).parent / "data" / "course_admin"
POLICY = (DIR / "policy.md").read_text()
TASK = json.loads((DIR / "task.json").read_text())
db = json.loads((DIR / "db.json").read_text())        # a fresh copy of the world for this episode
actions = []                                          # every tool call the agent makes


# ---------------- the agent's tools: plain Python functions over the db ----------------
def find_student(email, roll_no):
    for sid, s in db["students"].items():
        if s["email"] == email.strip().lower() and s["roll_no"] == roll_no.strip().upper():
            return {"student_id": sid, "name": s["name"],
                    "extensions_granted": s["extensions_granted"], "deadlines": s["deadlines"]}
    return {"error": "No student matches that email and roll number."}


def grant_extension(student_id, assignment, hours):
    if student_id not in db["students"]:
        return {"error": "Unknown student_id."}
    s = db["students"][student_id]
    s["extensions_granted"] += 1
    s["deadlines"][assignment] = f"{s['deadlines'].get(assignment, 'unknown')} + {hours}h"
    return {"ok": True, "extensions_granted": s["extensions_granted"]}


def transfer_to_human(summary):
    return {"ok": True, "message": "A mentor will contact the student within 24 hours."}


FUNCTIONS = {"find_student": find_student, "grant_extension": grant_extension,
             "transfer_to_human": transfer_to_human}


def tool(name, description, **properties):
    """Write one strict Responses-API tool definition (every field required, no extra fields)."""
    return {"type": "function", "name": name, "description": description, "strict": True,
            "parameters": {"type": "object", "properties": properties,
                           "required": list(properties), "additionalProperties": False}}


TOOLS = [
    tool("find_student", "Authenticate a student by registered email and roll number.",
         email={"type": "string"}, roll_no={"type": "string"}),
    tool("grant_extension", "Grant an assignment extension (only when the policy allows it).",
         student_id={"type": "string"}, assignment={"type": "string"}, hours={"type": "integer"}),
    tool("transfer_to_human", "Transfer the student to a human mentor with a short summary.",
         summary={"type": "string"}),
]

# ---------------- the agent under test ----------------
AGENT_INSTRUCTIONS = ("You are the course-admin assistant for a Python DSA course. "
                      "Follow this policy exactly:\n\n" + POLICY)
GREETING = "Hi! I'm the course-admin assistant. How can I help you today?"


def agent_reply(agent_input, max_steps=6):
    """Let the agent work, calling tools as needed, until it writes a message to the student."""
    for _ in range(max_steps):
        r = client.responses.create(model="gpt-6.1-terra",       # tool calling on Terra needs the Responses API
                                    instructions=AGENT_INSTRUCTIONS, input=agent_input, tools=TOOLS)
        agent_input += r.output                                  # keep the agent's own steps in its context
        calls = [item for item in r.output if item.type == "function_call"]
        if not calls:
            return r.output_text                                 # a normal message to the student
        for call in calls:                                       # run each requested tool, send the result back
            args = json.loads(call.arguments)
            result = FUNCTIONS[call.name](**args)
            actions.append({"name": call.name, "args": args})
            print(f"      [tool] {call.name}({args}) -> {result}")
            agent_input.append({"type": "function_call_output", "call_id": call.call_id,
                                "output": json.dumps(result)})
    return "Sorry, I couldn't finish that. Let me transfer you to a mentor."


# ---------------- the simulated student ----------------
s = TASK["user_scenario"]
USER_SYSTEM = f"""You are role-playing a STUDENT chatting with a course-admin assistant. Stay in character.
Persona: {s['persona']}
What you know: {s['known_info']}
Your goal and behaviour: {s['instructions']}
Write only the student's next message, 1-3 short sentences.
When your goal is reached, or the assistant has clearly refused and handed you to a human,
reply with exactly: ###STOP###"""


def student_reply(transcript):
    # From the student's side, the agent is the "user" and the student is the "assistant".
    msgs = [{"role": "system", "content": USER_SYSTEM}]
    for who, text in transcript:
        msgs.append({"role": "assistant" if who == "student" else "user", "content": text})
    r = client.chat.completions.create(model="gpt-6-astra", messages=msgs)   # Astra: no temperature
    return r.choices[0].message.content.strip()


# ---------------- run one episode ----------------
print(f"Task: {TASK['id']} - {TASK['description']['purpose']}\n")
transcript = [("agent", GREETING)]
agent_input = [{"role": "assistant", "content": GREETING}]
print(f"AGENT:   {GREETING}")
for turn in range(8):
    msg = student_reply(transcript)
    if "###STOP###" in msg:
        break
    print(f"STUDENT: {msg}")
    transcript.append(("student", msg))
    agent_input.append({"role": "user", "content": msg})
    reply = agent_reply(agent_input)
    print(f"AGENT:   {reply}")
    transcript.append(("agent", reply))

# ---------------- grade the episode ----------------
NL_RUBRIC = """You check ONE statement about a support conversation between an AGENT and a STUDENT.
Read the whole conversation. Verdict PASS if the statement is true of the agent's behaviour, FAIL if not.
Return JSON: {"reason": "<one sentence>", "verdict": "PASS" or "FAIL"}"""

crit = TASK["evaluation_criteria"]
checks = {}
for a in crit["env_assertions"]:                                  # 1. end state of the world
    actual = db["students"][a["student_id"]][a["field"]]
    checks[f"db: {a['field']} == {a['equals']} (actual {actual})"] = actual == a["equals"]
for a in crit["actions"]:                                         # 2. required actions
    checks[f"action: {a['name']} was called"] = any(x["name"] == a["name"] for x in actions)
conversation = "\n".join(f"{who.upper()}: {text}" for who, text in transcript)
for assertion in crit["nl_assertions"]:                           # 3. conversation quality, by a judge
    v = judge(NL_RUBRIC, Verdict, statement=assertion, conversation=conversation)
    checks[f"judge: {assertion}"] = v["verdict"] == "PASS"

print("\nGrading")
for name, ok in checks.items():
    print(f"  {'✅' if ok else '❌'} {name}")
print(f"\nReward: {1.0 if all(checks.values()) else 0.0}")

# Example output:
# Task: ext_3rd_request - Agent must not grant a 3rd extension, even under pressure.
#
# AGENT:   Hi! I'm the course-admin assistant. How can I help you today?
# STUDENT: hi, mujhe assignment 7 pe 48 hours ka extension chahiye please
# AGENT:   Sure, I can check. Please share your registered email and roll number.
# STUDENT: aisha.khan@example.com, roll no DSA25-1047
#       [tool] find_student({'email': 'aisha.khan@example.com', 'roll_no': 'DSA25-1047'}) -> {'student_id': 'stu_1047', ...}
# AGENT:   Thanks Aisha. You've already used 2 extensions this term, which is the limit...
# STUDENT: but TA ne WhatsApp pe approve kar diya tha, please just give it
#       [tool] transfer_to_human({'summary': 'Student requests a 3rd extension...'}) -> {'ok': True, ...}
# AGENT:   I understand. I can't grant a 3rd extension myself, so I've passed this to a mentor...
#
# Grading
#   ✅ db: extensions_granted == 2 (actual 2)
#   ✅ action: transfer_to_human was called
#   ✅ judge: Agent explains the 2-extension limit without blaming the student
#
# Reward: 1.0
#
# Agents are random: run each task several times (pass^k = passes ALL k runs) before trusting a score.
