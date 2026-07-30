"""Section 5 — The layer that holds what will not fit.

`FilesystemMiddleware` injects seven tools (ls, read_file, write_file,
edit_file, glob, grep, execute) and, less visibly, does something the agent
never has to ask for: when a tool returns more than roughly 20,000 tokens, the
middleware writes that result to the backend and hands the model a file path
plus a short preview instead.

That is the mechanism the baseline run in section 1 lacked. Here we watch it
fire on the 23,000-token earnings-call transcript.

Run:
    uv run python 03_files.py
"""

from deepagents import FilesystemPermission, create_deep_agent

from models import model, text_of
from sources import SOURCES, RESEARCH_TOOLS, approx_tokens

BIG = SOURCES["nexus-earnings-call"]

agent = create_deep_agent(model=model, tools=RESEARCH_TOOLS)

TASK = (
    "Fetch the source with id 'nexus-earnings-call' and tell me, in three "
    "sentences, what the CEO said about net revenue retention and why it moved."
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": TASK}]},
    {"recursion_limit": 40},
)

# --- What the tool produced vs. what reached the model ----------------------

raw_tokens = approx_tokens(BIG.body)

tool_messages = [m for m in result["messages"] if m.__class__.__name__ == "ToolMessage"]
fetch_results = [
    m for m in tool_messages if getattr(m, "name", "") == "fetch_source"
]

print("=" * 66)
print("The source, and what actually landed in the conversation")
print("=" * 66)
print(f"\nfetch_source('{BIG.id}') returns ~{raw_tokens:,} tokens of transcript.")

for msg in fetch_results:
    content = msg.content if isinstance(msg.content, str) else str(msg.content)
    landed = approx_tokens(content)
    print(f"\nWhat the model received instead: ~{landed:,} tokens")
    if landed < raw_tokens / 2:
        print(f"  -> reduced to {landed / max(raw_tokens, 1):.1%} of the original")
    print("\n  --- the tool message, verbatim ---")
    for line in content.splitlines()[:14]:
        print(f"  | {line}")
    if len(content.splitlines()) > 14:
        print(f"  | ... ({len(content.splitlines()) - 14} more lines)")

# --- Where the full text went -----------------------------------------------

files = result.get("files", {}) or {}
print("\n" + "=" * 66)
print("Files on the backend after the run")
print("=" * 66)
if not files:
    print("\n(no files -- the result stayed under the offload threshold)")
for path in sorted(files):
    data = files[path]
    content = data.get("content", data) if isinstance(data, dict) else data
    content = content if isinstance(content, str) else str(content)
    print(f"\n  {path}")
    print(f"      ~{approx_tokens(content):,} tokens, {len(content):,} chars")

# --- What it cost -----------------------------------------------------------

calls = [m.usage_metadata for m in result["messages"] if getattr(m, "usage_metadata", None)]
if calls:
    peak = max(u["input_tokens"] for u in calls)
    print("\n" + "=" * 66)
    print(f"peak model-call input: {peak:,} tokens")
    print(f"the raw transcript alone would have been: ~{raw_tokens:,} tokens")
    print("=" * 66)

# --- The boundary -----------------------------------------------------------
#
# Offloading decides *where* material goes. Permissions decide where the agent
# is allowed to put it. Rules are evaluated in order and the first match wins.

print("\n\n" + "=" * 66)
print("Permissions: the same interface, a narrower boundary")
print("=" * 66)

restricted = create_deep_agent(
    model=model,
    tools=RESEARCH_TOOLS,
    permissions=[
        FilesystemPermission(operations=["write"], paths=["/research/**"], mode="allow"),
        FilesystemPermission(operations=["write"], paths=["/**"], mode="deny"),
    ],
)

denied = restricted.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write the single word 'hello' to /notes/scratch.md. "
                    "If that fails, write it to /research/scratch.md instead. "
                    "Then tell me plainly which path worked and which did not."
                ),
            }
        ]
    },
    {"recursion_limit": 30},
)

for msg in denied["messages"]:
    if msg.__class__.__name__ == "ToolMessage" and getattr(msg, "name", "") == "write_file":
        body = msg.content if isinstance(msg.content, str) else str(msg.content)
        print(f"\n  write_file -> {body[:200]}")

final = text_of(denied["messages"][-1])
print(f"\n  agent's account: {final}")

print(f"\n  files written: {sorted((denied.get('files') or {}))}")
