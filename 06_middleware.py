"""Section 8 — Writing your own layer.

Everything the chapter has shown so far is a layer somebody else wrote. This is
where you write two of your own, using the two hook styles.

`wrap_tool_call` is a wrap-style hook: it receives the call and the handler, and
decides when — or whether — to invoke it. That makes it the place for anything
that needs to see both sides of a tool call: timing, size, retries, caching,
redaction.

`before_model` is a node-style hook: it runs at a fixed point and returns a
state update, or a jump. That makes it the place for anything that has to
decide whether the run should continue at all.

Run:
    uv run python 06_middleware.py
"""

import time

from langchain.agents.middleware import before_model, wrap_tool_call
from langchain.messages import AIMessage
from deepagents import create_deep_agent

from models import model, text_of
from sources import BRIEF_TASK, RESEARCH_TOOLS

# --- Layer one: a ledger ----------------------------------------------------

LEDGER: list[dict] = []


@wrap_tool_call
def ledger(request, handler):
    """Record what every tool call cost, on both sides of the call."""
    name = request.tool_call["name"]
    started = time.monotonic()
    result = handler(request)
    elapsed = time.monotonic() - started

    content = getattr(result, "content", result)
    size = len(content) if isinstance(content, str) else len(str(content))

    LEDGER.append({"tool": name, "seconds": elapsed, "chars": size})
    return result


# --- Layer two: a budget guard ----------------------------------------------
#
# The harness will summarize when the context approaches the model's limit. That
# protects the model. It does not protect your bill. This stops the run.

TOKEN_CEILING = 60_000
SPENT = {"input": 0}


@before_model(can_jump_to=["end"])
def budget_guard(state, runtime):
    """Halt the run once cumulative input tokens cross a ceiling."""
    spent = sum(
        m.usage_metadata["input_tokens"]
        for m in state["messages"]
        if getattr(m, "usage_metadata", None)
    )
    SPENT["input"] = spent
    if spent >= TOKEN_CEILING:
        return {
            "messages": [
                AIMessage(
                    f"Stopping: this run has consumed {spent:,} input tokens, "
                    f"which is over the {TOKEN_CEILING:,} ceiling set for it."
                )
            ],
            "jump_to": "end",
        }
    return None


agent = create_deep_agent(
    model=model,
    tools=RESEARCH_TOOLS,
    middleware=[ledger, budget_guard],
)

print("Both layers are in the stack:")
print(f"  {[n for n in agent.get_graph().nodes if not n.startswith('__')]}\n")
print("Note that only budget_guard appears as a node. ledger uses a wrap-style")
print("hook, so it wraps the tools node rather than sitting beside it.\n")

result = agent.invoke(
    {"messages": [{"role": "user", "content": BRIEF_TASK}]},
    {"recursion_limit": 60},
)

# --- What the ledger saw ----------------------------------------------------

print("=" * 66)
print(f"{'tool':<22} {'calls':>6} {'total s':>9} {'total chars':>13}")
print("=" * 66)

by_tool: dict[str, dict] = {}
for entry in LEDGER:
    agg = by_tool.setdefault(entry["tool"], {"calls": 0, "seconds": 0.0, "chars": 0})
    agg["calls"] += 1
    agg["seconds"] += entry["seconds"]
    agg["chars"] += entry["chars"]

for name, agg in sorted(by_tool.items(), key=lambda kv: -kv[1]["chars"]):
    print(f"{name:<22} {agg['calls']:>6} {agg['seconds']:>9.2f} {agg['chars']:>13,}")

print("=" * 66)
print(f"{'TOTAL':<22} {len(LEDGER):>6} "
      f"{sum(e['seconds'] for e in LEDGER):>9.2f} "
      f"{sum(e['chars'] for e in LEDGER):>13,}")

print(f"\nbudget guard: {SPENT['input']:,} / {TOKEN_CEILING:,} input tokens used")
halted = "Stopping: this run has consumed" in text_of(result["messages"][-1])
print(f"run halted by the guard: {'yes' if halted else 'no'}")

print(
    "\nNeither layer changed what the agent can do. They changed what you can\n"
    "see and what you can stop -- which is most of what running an agent in\n"
    "production actually consists of."
)
