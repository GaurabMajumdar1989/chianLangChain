"""04 · The filesystem for intermediate state.

The third knob — except you don't even turn it: the file tools ship by default,
just like the planning tool. A deep agent can `write_file`, `read_file`, `ls`,
`edit_file`, `glob`, and `grep`. The point from the chapter is context
engineering: bulky intermediate results get written to files and pulled back only
when needed, instead of accumulating in the model's context window and rotting it.

By default this filesystem is backed by the agent's STATE, not your real disk —
an ephemeral, virtualized directory. Nothing is written to your machine. After the
run we print `result["files"]` to reveal that virtual filesystem: the artifacts
the agent parked there to keep them out of its own context.

In production you'd swap the backend (a sandbox, or a durable store like Firestore
or DynamoDB) and use path-based FilesystemPermission rules to scope access — the
interface stays the same. Needs only OPENAI_API_KEY.
"""

from deepagents import create_deep_agent

from models import model

agent = create_deep_agent(model=model)

TASK = (
    "Do this using your file tools, not your context:\n"
    "1. Write three files under /notes/ — planning.md, subagents.md, "
    "filesystem.md — each with a two-sentence description of that deep-agent "
    "capability.\n"
    "2. Then read the three files back and write a combined /summary.md that "
    "lists all three capabilities.\n"
    "3. Reply with only the contents of /summary.md."
)

result = agent.invoke({"messages": [{"role": "user", "content": TASK}]})

print("=== Final answer (contents of /summary.md) ===")
print(result["messages"][-1].content)

# --- Reveal the virtual filesystem -----------------------------------------
# These files lived in agent state, never on your disk, and never fully in the
# model's context at once. This is the filesystem acting as a context engine.
print("\n=== Files in agent state (result['files']) ===")
files = result.get("files", {})
for path in sorted(files):
    data = files[path]
    # Each entry is a FileData dict ({"content": ..., "encoding": ...}); older
    # backends may store the content as a plain string.
    content = data.get("content", data) if isinstance(data, dict) else data
    print(f"\n----- {path} -----")
    print(content)
