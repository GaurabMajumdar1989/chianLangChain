"""01 · The bare deep agent.

The on-ramp for the chapter. `create_deep_agent` returns a fully assembled harness
from a single argument — the model. You configured nothing: no tools=, no
backend=, no subagents=, no system_prompt=. Yet this one-line agent already
carries a planning tool, a filesystem interface, an `execute` tool, the `task`
tool for delegation, and the base system prompt. That is the harness doing the
heavy lifting.

Two details carry through every later example:
  1. The invoke contract — you pass a message list, not a bare string, and you
     read the answer off the LAST message of the returned list.
  2. You configured nothing — each following example turns exactly one knob on
     this same create_deep_agent call.

Run it to confirm your setup before moving on. Needs only OPENAI_API_KEY.
"""

from deepagents import create_deep_agent

from models import model

agent = create_deep_agent(model=model)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is an LLM?"}]}
)

print(result["messages"][-1].content)
