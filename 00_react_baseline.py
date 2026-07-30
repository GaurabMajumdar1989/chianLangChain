"""Section 1 — A run that falls over.

The chapter opens with a measurement rather than a definition. This is that
measurement: the competitive-brief task handed to a plain tool-calling agent
with no harness around it, instrumented so you can watch the context grow.

The agent is not badly written and the model is not weak. What goes wrong is
structural: the corpus contains one source of roughly 23,000 tokens, the agent
has no way to hold it anywhere except the conversation, and so every model call
after it reads that source carries it again.

Run:
    uv run python 00_react_baseline.py
"""

import time

from langchain.agents import create_agent

from models import model, text_of
from sources import BRIEF_TASK, RESEARCH_TOOLS

agent = create_agent(model=model, tools=RESEARCH_TOOLS)

print("Running the brief task on a plain tool-calling agent...\n")
started = time.monotonic()
result = agent.invoke(
    {"messages": [{"role": "user", "content": BRIEF_TASK}]},
    {"recursion_limit": 40},
)
elapsed = time.monotonic() - started

# --- What the provider says it was actually sent ----------------------------
#
# Every AI message carries usage_metadata reported by the provider. These are
# not estimates; they are the tokens billed for that call.

calls = [
    m.usage_metadata
    for m in result["messages"]
    if getattr(m, "usage_metadata", None)
]

print("=" * 62)
print(f"{'model call':>10} {'input tokens':>14} {'output':>9}   growth")
print("=" * 62)

first_input = calls[0]["input_tokens"] if calls else 0
for i, usage in enumerate(calls, start=1):
    tokens_in = usage["input_tokens"]
    bar = "#" * max(1, round(tokens_in / 900))
    print(f"{i:>10} {tokens_in:>14,} {usage['output_tokens']:>9,}   {bar}")

peak = max((u["input_tokens"] for u in calls), default=0)
billed = sum(u["input_tokens"] for u in calls)

print("=" * 62)
print(f"model calls           : {len(calls)}")
print(f"messages in history   : {len(result['messages'])}")
print(f"first call input      : {first_input:,} tokens")
print(f"peak call input       : {peak:,} tokens")
print(f"growth first -> peak  : {peak / max(first_input, 1):.1f}x")
print(f"total input tokens    : {billed:,} (this is what you pay for)")
print(f"wall clock            : {elapsed:.1f}s")

# --- Did it actually produce the four sections it was asked for? ------------

answer = text_of(result["messages"][-1])

print("\n" + "=" * 62)
print("Did the deliverable survive?")
print("=" * 62)
for heading in ["market shape", "vendor", "case against", "recommend"]:
    mark = "yes" if heading in answer.lower() else "NO "
    print(f"  section mentioning {heading!r}: {mark}")
print(f"\nfinal answer length: {len(answer):,} chars")
print("\n--- final answer ---")
print(answer)
