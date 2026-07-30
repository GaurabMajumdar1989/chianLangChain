"""Section 3 — The harness as a stack.

The claim this chapter rests on is that `create_deep_agent` is not a black box
with four magic properties. It is `create_agent` plus an ordered list of
middleware, and you can read that list off the object you get back.

This lab makes no model calls. It builds two agents and inspects them, so it
runs without spending anything and without a working API key doing real work.

Run:
    uv run python 01_stack.py
"""

from langchain.agents import create_agent
from deepagents import create_deep_agent

from models import model
from sources import RESEARCH_TOOLS


def tool_names(agent) -> list[str]:
    """Read the tools an assembled agent actually ended up with.

    The compiled graph has a node named "tools" holding a ToolNode, and that
    node knows every tool the model will be offered — including the ones no
    caller passed in.
    """
    node = agent.get_graph().nodes.get("tools")
    if node is None:
        return []
    return sorted(node.data.tools_by_name)


def graph_nodes(agent) -> list[str]:
    return [n for n in agent.get_graph().nodes if not n.startswith("__")]


# --- Two agents, the same tools passed in -----------------------------------

plain = create_agent(model=model, tools=RESEARCH_TOOLS)
deep = create_deep_agent(model=model, tools=RESEARCH_TOOLS)

plain_tools = tool_names(plain)
deep_tools = tool_names(deep)

print("=" * 68)
print("We passed the SAME two tools to both agents.")
print("=" * 68)
print(f"\ncreate_agent      -> {len(plain_tools)} tools: {', '.join(plain_tools)}")
print(f"create_deep_agent -> {len(deep_tools)} tools: {', '.join(deep_tools)}")

free = [t for t in deep_tools if t not in plain_tools]
print(f"\n{len(free)} tools appeared that nobody passed in:")

# Which middleware injected each one. This mapping is not guesswork: it is the
# order the stack is assembled in, in deepagents/graph.py.
INJECTED_BY = {
    "write_todos": "TodoListMiddleware",
    "ls": "FilesystemMiddleware",
    "read_file": "FilesystemMiddleware",
    "write_file": "FilesystemMiddleware",
    "edit_file": "FilesystemMiddleware",
    "glob": "FilesystemMiddleware",
    "grep": "FilesystemMiddleware",
    "execute": "FilesystemMiddleware",
    "task": "SubAgentMiddleware",
}
for name in free:
    print(f"    {name:<12} <- {INJECTED_BY.get(name, 'unknown')}")

# --- The stack is not a metaphor; some of it is literally graph nodes -------

print("\n" + "=" * 68)
print("Middleware that uses node-style hooks becomes a node in the graph.")
print("=" * 68)
print(f"\ncreate_agent graph nodes:      {graph_nodes(plain)}")
print(f"create_deep_agent graph nodes: {graph_nodes(deep)}")
print(
    "\nMiddleware using wrap-style hooks (FilesystemMiddleware,\n"
    "SubAgentMiddleware, SummarizationMiddleware) does not appear here --\n"
    "it wraps the model and tool calls rather than sitting beside them.\n"
    "Its fingerprint is the injected tools above."
)

# --- Adding your own layer shifts nothing else ------------------------------

from langchain.agents.middleware import before_model  # noqa: E402


@before_model
def noop(state, runtime):
    """A layer that does nothing, purely to show where it lands."""
    return None


with_extra = create_deep_agent(model=model, tools=RESEARCH_TOOLS, middleware=[noop])

print("\n" + "=" * 68)
print("Passing middleware= inserts your layer into the same stack.")
print("=" * 68)
print(f"\nwith middleware=[noop] -> {graph_nodes(with_extra)}")
print(
    "\nYour layer is merged in at a defined position: after the harness's own\n"
    "core middleware, and before the profile, prompt-caching and memory\n"
    "layers. That ordering is deliberate -- prompt caching has to run last so\n"
    "the cached prefix matches what is actually sent to the model."
)
