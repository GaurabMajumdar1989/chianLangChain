"""02 · The planning tool (write_todos).

The first knob. We add NOTHING to the create_deep_agent call from example 01 —
the planning tool ships by default. What changes is the task: give a deep agent a
job with several dependent deliverables and it stops guessing its way forward and
instead externalizes a plan.

Under the hood the agent calls a built-in `write_todos` tool. Each todo is a
structured item — `content` plus a `status` of pending / in_progress / completed —
and the agent rewrites the whole list between steps as work progresses. A UI would
render that as a checklist; here we read it straight off the message history so you
can see the durable, structured contract the chapter describes.

Needs only OPENAI_API_KEY.
"""

from deepagents import create_deep_agent

from models import model

# Same bare agent as example 01 — the planning tool is already on board.
agent = create_deep_agent(model=model)

# The planning tool is OPTIONAL — the agent decides whether a task is worth
# planning, and for a task it deems simple it may skip write_todos entirely and
# just answer. To make the plan reliably visible, we explicitly ask it to use the
# tool and work through the todos one at a time.
TASK = (
    "Use your write_todos planning tool for this task. FIRST call write_todos to "
    "lay out a plan with one todo per section, then work through them one at a "
    "time, updating each todo's status (in_progress, then completed) as you go.\n\n"
    "Deliverable: a short internal briefing titled 'Should we adopt Deep Agents?' "
    "with exactly three sections: (1) what problem harnesses solve, (2) the four "
    "capabilities that make an agent deep, and (3) a recommendation."
)

result = agent.invoke({"messages": [{"role": "user", "content": TASK}]})


# --- Make the plan visible -------------------------------------------------
# Walk the message history and print every write_todos call. Watching the list
# evolve across calls shows the agent marking tasks pending -> in_progress ->
# completed, rather than holding the plan implicitly in its reasoning.
def print_plans(messages):
    step = 0
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            if call["name"] != "write_todos":
                continue
            step += 1
            print(f"\n--- Plan update #{step} ---")
            for todo in call["args"].get("todos", []):
                mark = {"completed": "[x]", "in_progress": "[~]"}.get(
                    todo.get("status"), "[ ]"
                )
                print(f"  {mark} {todo.get('content')}")
    return step


plan_updates = print_plans(result["messages"])
if plan_updates == 0:
    # The agent chose not to plan (it can, for tasks it deems simple). Nothing
    # went wrong — there is just no plan to show. Try a more complex task.
    print("(The agent answered without calling the planning tool this time.)")

print("\n=== Final briefing ===")
print(result["messages"][-1].content)
