"""Section 4 — The layer that plans.

`TodoListMiddleware` is the first entry in the default stack. It injects one
tool, `write_todos`, and appends guidance telling the model when to use it.
That is the whole mechanism.

The point worth taking from this lab is that the plan is *state*, not prose.
The agent does not describe its plan in a sentence you have to parse. It calls
a tool whose arguments are a structured list, which means the plan is something
your code can read, render, validate, or store — as this script does.

Run:
    uv run python 02_plan.py
"""

from deepagents import create_deep_agent

from models import model
from sources import BRIEF_TASK, RESEARCH_TOOLS

# Nothing here configures planning. It arrives with the harness.
agent = create_deep_agent(model=model, tools=RESEARCH_TOOLS)

result = agent.invoke(
    {"messages": [{"role": "user", "content": BRIEF_TASK}]},
    {"recursion_limit": 60},
)

# --- Read the plan off the tool calls ---------------------------------------

MARKS = {"completed": "[x]", "in_progress": "[~]", "pending": "[ ]"}

revisions = [
    call["args"].get("todos", [])
    for message in result["messages"]
    for call in (getattr(message, "tool_calls", None) or [])
    if call["name"] == "write_todos"
]

if not revisions:
    print(
        "The agent judged this task simple enough to skip planning.\n"
        "That is the middleware exercising judgement, not a failure -- see the\n"
        "note at the end of this section in the chapter."
    )
else:
    print(f"The plan was rewritten {len(revisions)} times.\n")
    for n, todos in enumerate(revisions, start=1):
        done = sum(1 for t in todos if t.get("status") == "completed")
        print(f"--- revision {n}  ({done}/{len(todos)} complete) ---")
        for todo in todos:
            mark = MARKS.get(todo.get("status"), "[ ]")
            print(f"  {mark} {todo.get('content')}")
        print()

    # The plan is data. Prove it by using it as data.
    final = revisions[-1]
    print("=" * 60)
    print("The final revision, as the structure it actually is:")
    print("=" * 60)
    for todo in final:
        print(f"  {todo!r}")

print("\n" + "=" * 60)
print(f"model calls in this run: "
      f"{sum(1 for m in result['messages'] if getattr(m, 'usage_metadata', None))}")
print("=" * 60)
