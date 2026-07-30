"""Section 6 — The layer that delegates.

`SubAgentMiddleware` injects one tool, `task`, and the argument for it is an
accounting argument rather than an architectural one. A subagent runs its own
loop, in its own message history, and returns one string. Everything it read to
produce that string is billed once, inside the subagent, and never enters the
coordinator's context.

This lab makes both halves of that visible: the coordinator has no research
tools at all, so it *must* delegate, and we count what each side spent.

Run:
    uv run python 04_delegate.py
"""

from deepagents import create_deep_agent
from langchain_core.tools import tool

from models import model, text_of
from sources import SOURCES, approx_tokens, fetch_source, search_sources

# --- Instrument the tools so we can see who called them ---------------------

CALL_LOG: list[tuple[str, str]] = []


@tool
def search_sources_logged(query: str) -> str:
    """Search the research corpus for sources relevant to a query."""
    CALL_LOG.append(("search_sources", query))
    return search_sources.invoke({"query": query})


@tool
def fetch_source_logged(source_id: str) -> str:
    """Fetch the full text of one source by its id."""
    CALL_LOG.append(("fetch_source", source_id))
    return fetch_source.invoke({"source_id": source_id})


# --- The specialist ---------------------------------------------------------
#
# `description` is not documentation for you. It is what the coordinator reads
# when it decides whether to delegate and what to send. `system_prompt` is the
# subagent's own brain and is never inherited from the parent. `tools` replaces
# rather than extends what the parent holds.

analyst = {
    "name": "source-analyst",
    "description": (
        "Reads ONE source from the research corpus and returns a single "
        "paragraph of findings relevant to a stated question. Delegate one "
        "source id per call. Use this for every source you need read."
    ),
    "system_prompt": (
        "You read exactly one source and report on it.\n"
        "Call fetch_source_logged with the source id you were given, read it, "
        "and return ONE paragraph of at most 90 words covering only what that "
        "source supports.\n"
        "State figures precisely. If the source undercuts its own claim — a "
        "methodology caveat, a self-reported number, a small sample — say so "
        "in the same paragraph. Do not speculate beyond the text."
    ),
    "tools": [fetch_source_logged],
}

# --- The coordinator --------------------------------------------------------
#
# Note what it is NOT given: no fetch_source. It can see the index and nothing
# more. The only route to a source's contents is through the specialist.

COORDINATOR = (
    "You are a research coordinator. You cannot read sources yourself — you "
    "have no tool that returns source text.\n"
    "Work in two steps. First call search_sources_logged once to see what "
    "exists. Then delegate EVERY source you want read to the source-analyst "
    "subagent using the task tool, one source id per delegation, telling it "
    "what question to answer.\n"
    "Finally, synthesise the returned paragraphs into a four-section brief: "
    "market shape, the vendors, the case against a specialized engine, and a "
    "recommendation. Where the returned findings disagree, say so rather than "
    "splitting the difference."
)

agent = create_deep_agent(
    model=model,
    tools=[search_sources_logged],
    system_prompt=COORDINATOR,
    subagents=[analyst],
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Produce the brief. Read at least four sources, and include "
                    "'nexus-earnings-call' among them."
                ),
            }
        ]
    },
    {"recursion_limit": 80},
)

# --- Who called what --------------------------------------------------------

print("=" * 66)
print("Tool calls, in order")
print("=" * 66)
for name, arg in CALL_LOG:
    where = "coordinator" if name == "search_sources" else "  subagent  "
    print(f"  [{where}] {name}({arg!r})")

fetches = [a for n, a in CALL_LOG if n == "fetch_source"]
read_tokens = sum(approx_tokens(SOURCES[a].body) for a in fetches if a in SOURCES)

# --- The accounting ---------------------------------------------------------

delegations = [
    call
    for message in result["messages"]
    for call in (getattr(message, "tool_calls", None) or [])
    if call["name"] == "task"
]

returned = [
    m.content if isinstance(m.content, str) else str(m.content)
    for m in result["messages"]
    if m.__class__.__name__ == "ToolMessage" and getattr(m, "name", "") == "task"
]
returned_tokens = sum(approx_tokens(c) for c in returned)

calls = [m.usage_metadata for m in result["messages"] if getattr(m, "usage_metadata", None)]
peak = max((u["input_tokens"] for u in calls), default=0)

print("\n" + "=" * 66)
print("What crossed the boundary")
print("=" * 66)
print(f"\n  delegations                    : {len(delegations)}")
print(f"  sources read (inside subagents): {len(fetches)}")
print(f"  tokens of source text read     : ~{read_tokens:,}")
print(f"  tokens returned to coordinator : ~{returned_tokens:,}")
if read_tokens:
    print(f"  crossed the boundary           : {returned_tokens / read_tokens:.1%}")
print(f"\n  coordinator peak call input    : {peak:,} tokens")
print(
    "\n  The coordinator never saw the transcript. It saw a paragraph about it."
)

print("\n" + "=" * 66)
print("The brief")
print("=" * 66)
answer = text_of(result["messages"][-1])
print(answer)
