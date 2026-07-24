"""03 · Subagents and hierarchical delegation.

The second knob: `subagents=`. A deep agent can spawn specialized workers, each
with its OWN system prompt and its OWN tools, that run in an isolated context and
return only their final result — not their intermediate reasoning. This is the
delegation pattern from the chapter: the main agent hands off a scoped job, the
subagent does the messy work in its own context window, and the main agent's
context stays clean.

Here a coordinator delegates individual topic lookups to a `fact-researcher`
subagent. The researcher owns the lookup tool; the coordinator does not. It can
only get facts by delegating through the built-in `task` tool. We use a small
in-memory knowledge base so the example runs with no key beyond OPENAI_API_KEY.
"""

from deepagents import create_deep_agent
from langchain_core.tools import tool

from models import model

# --- The subagent's private tool -------------------------------------------
# A stand-in for a real search / database tool. It belongs ONLY to the
# researcher subagent (see `tools=` below), so the coordinator cannot call it.
_KNOWLEDGE_BASE = {
    "planning tool": "Deep agents externalize a structured todo list (write_todos) "
    "with per-item status, updated between steps, instead of planning implicitly.",
    "subagents": "Deep agents spawn specialized workers with their own prompt and "
    "tools that run in an isolated context and return only a final result.",
    "filesystem": "Deep agents write intermediate artifacts to a virtual filesystem "
    "so bulky material stays out of the model's context window.",
    "system prompt": "Deep agents rely on a large, curated system prompt that "
    "encodes identity, scope, a reasoning framework, and heuristics.",
}


@tool
def lookup_fact(topic: str) -> str:
    """Look up a factual summary about a deep-agent topic from the knowledge base.
    Recognizes any topic that contains a known keyword (e.g. 'the filesystem
    capability' matches 'filesystem')."""
    print(f"    >> [researcher] lookup_fact(topic='{topic}')")
    key = topic.lower().strip()
    for name, fact in _KNOWLEDGE_BASE.items():
        if name in key or key in name:
            return fact
    return f"No entry found for '{topic}'."


# --- The specialized subagent ----------------------------------------------
# Note the key is `system_prompt` (its own brain, never inherited) and `tools`
# overrides the inherited set with just the lookup tool.
fact_researcher = {
    "name": "fact-researcher",
    "description": (
        "Look up a factual summary for ONE deep-agent topic and return a single "
        "polished sentence. Delegate one topic per call."
    ),
    "system_prompt": (
        "You research one topic at a time. Call lookup_fact with the given "
        "topic, then return ONE clear sentence based only on what it returns. "
        "Do not add facts of your own."
    ),
    "tools": [lookup_fact],
}


# --- The coordinator (main agent) ------------------------------------------
COORDINATOR_PROMPT = (
    "You are a coordinator. You do NOT look up facts yourself — you have no "
    "lookup tool. For each topic the user asks about, delegate to the "
    "fact-researcher subagent using the task tool (one topic per delegation). "
    "Collect the returned sentences and combine them into a short bulleted "
    "summary. Keep your own context focused on coordination."
)

agent = create_deep_agent(
    model=model,
    system_prompt=COORDINATOR_PROMPT,
    subagents=[fact_researcher],
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Summarize two deep-agent capabilities: subagents "
                "and the filesystem.",
            }
        ]
    }
)

print("\n=== Coordinator's assembled summary ===")
print(result["messages"][-1].content)
