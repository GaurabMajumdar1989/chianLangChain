import os
import json
import re
import inspect
from dotenv import load_dotenv

load_dotenv()
from openai import OpenAI
from langsmith import traceable

MAX_ITERATIONS = 5
MODEL = "nvidia/nemotron-nano-9b-v2:free"

openai_client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY"),
)

print("HEllo Mysterious Entity......!!!!")



@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """
        Look at the price of the product in the catalog.
    """
    print(f">> Fetching price of the product {product} ")
    prices={"laptop":1244.45, "headphones": 980, "keyboard":25, "mouse":45.85}
    return prices.get(product, 0)

@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """ Apply a discount tier to the price and return the final price.
        Available tiers: bronze, silver and gold.
    """
    price=float(price)
    print(f" Applying discount on price:{price} with discount_tier:{discount_tier}")
    discount_percentages={"bronze": 3, "silver": 15, "gold": 27}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount/100), 2)

tools={
        "get_product_price":get_product_price,
        "apply_discount":apply_discount
    }

def get_tool_description(tools_dict):
    descriptions=[]
    for tool_name, tool_fnc in tools_dict.items():
        # __wrapped__ bypasses decorator wrappers (e.g traceable adds *, config=None)
        original_fnc = getattr(tool_fnc, "__wrapped__", tool_fnc)
        signature = inspect.signature(original_fnc)
        docstring = inspect.getdoc(original_fnc) or  "" 
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(descriptions)       

tool_descriptions=get_tool_description(tools)    
tool_names=", ".join(tools.keys())

react_prompt=f"""
STRICT RULES - You must follow these exactly:

1. NEVER guess or assume any product price.
   You MUST call get_product_price first to get the real price.

2. ONLY call apply_discount AFTER you have received a price
   from get_product_price.

3. When calling apply_discount, the FIRST Action Input argument
   MUST be the EXACT numeric price returned by get_product_price.
   NEVER pass the product name to apply_discount.

4. For example, if get_product_price returns:
   Observation: 540.51

   and the discount tier is silver, you MUST call:

   Action: apply_discount
   Action Input: 540.51, silver

5. NEVER invent or estimate an Observation.
   The Python runtime provides all Observations.

6. ALWAYS use apply_discount to calculate the discount.
   NEVER calculate the discount yourself.

7. If the user does not specify discount_tier,
   ask them which tier to use - NEVER assume one by yourself.


Use the following format:

Question: the input question you must answer
Thought: decide what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action

IMPORTANT:
- STOP immediately after Action Input.
- DO NOT generate Observation.
- DO NOT generate another Thought after Action Input.
- DO NOT generate another Action after Action Input.
- The Python runtime will execute the tool and provide the Observation.
- After the runtime provides an Observation, you will be called again.
- Use the Observation from the Python runtime when deciding your next Action.
- Only generate Final Answer when the available Observations are sufficient to answer the question.

Begin!

Question: {{question}}
Thought:
"""


def call_llm(model,msgs,options):
    resp = openai_client.chat.completions.create(
        model=model,
        messages=msgs,
        **options
    )
    return resp.choices[0].message.content

@traceable(name="Raw Agent Prompt")
def run_agent_raw_prompt(question: str):

    print(f"Question: {question}")
    print("="*90)

    prompt = react_prompt.format(question=question)
    scratchpad = ""

    for iteration in range(1, MAX_ITERATIONS+1):
        print(f"Iteration : ---- {iteration} ----")
        
        full_prompt = prompt + scratchpad
        options_for_llm={"stop":"\nObservation", "temperature":0}

        message_to_llm=[{"role":"user", "content":full_prompt}]

        output=call_llm(MODEL,message_to_llm,options_for_llm)

        print(f"OutPut=================\n\n{output}")

        print(f"  [Parsing] Looking for Final Answer in LLM output...")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f"  [Parsed] Final Answer: {final_answer}")
            print("\n" + "=" * 60)
            print(f"Final Answer: {final_answer}")
            return final_answer
        
         # CHANGE 6: Parse tool calls from raw text with regex — fragile if LLM doesn't follow format.
        print(f"  [Parsing] Looking for Action and Action Input in LLM output...")

        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match or not action_input_match:
            print(
                "  [Parsing] ERROR: Could not parse Action/Action Input from LLM output"
            )
            break

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        print(f"  [Tool Selected] {tool_name} with args: {tool_input_raw}")

        # Split comma-separated args; strip key= prefix if LLM outputs key=value format
        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]

        print(f"  [Tool Executing] {tool_name}({args})...")
        if tool_name not in tools:
            observation = f"Error: Tool '{tool_name}' not found. Available tools: {list(tools.keys())}"
        else:
            observation = str(tools[tool_name](*args))


        print(f"  [Tool Result] {observation}")

        # CHANGE 7: History is one growing string re-sent every iteration (replaces messages.append).
        scratchpad += f"{output}\nObservation: {observation}\nThought:"
        print(f"Iterative outputs: \n{scratchpad}")


    print("Error! Maxed out iterations.")
    return None    


if __name__ == "__main__":
    run_agent_raw_prompt(
        "What is the price of headphones with silver discount?"
    )