"""05 · The system prompt.

The last knob: `system_prompt=`. For a deep agent the system prompt is the
architecture of its reasoning, identity, and boundaries — it AUGMENTS the harness's
built-in base prompt (the one you got for free in example 01). This example writes
one that follows the five principles from the chapter:

  1. Clear identity and scope — what the agent is, and what it is NOT.
  2. Empower, don't constrain — give the end goal, not a fixed tool sequence.
  3. A reasoning framework, not a flowchart — a repeatable approach, not If/Then.
  4. Heuristic boundaries — compressed principles that cover many edge cases.
  5. Language efficiency — no repetition, no contradictory instructions.

Ask the same agent an in-scope and an out-of-scope question to see the prompt
shape its behavior. Needs only OPENAI_API_KEY.
"""

from deepagents import create_deep_agent

from models import model

SYSTEM_PROMPT = """\
You are OrderDesk, a support assistant for a hardware store's ONLINE ORDERS.

## Identity and scope
- You help with existing orders: status, changes, cancellations, returns, and
  delivery questions.
- You are NOT sales, and NOT marketing. You do not recommend products to buy or
  run promotions. If asked, say so briefly and redirect to what you can help with.

## How you work (a framework, not a script)
1. Identify the core issue in the customer's message.
2. Gather the minimum context you need to act — ask for an order number only if
   the request actually requires one.
3. Resolve: give the clearest next step you can.
4. Confirm the customer has what they need before closing.

## Operating principles
- Prefer the simplest resolution that fully solves the problem.
- One clarifying question is fine; a wall of questions is not.
- Never invent order details, prices, or policies you weren't given. If you don't
  know, say what you'd need to find out.
- Be concise and direct. No preamble, no upselling.
"""

agent = create_deep_agent(model=model, system_prompt=SYSTEM_PROMPT)

for question in [
    "My order hasn't arrived and it's been two weeks. What can I do?",
    "Which laptop should I buy for gaming?",
]:
    print(f"\n=== Customer: {question} ===")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )
    print(result["messages"][-1].content)
