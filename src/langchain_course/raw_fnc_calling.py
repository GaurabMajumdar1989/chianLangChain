import os
import json
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

@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """
        Look at the price of the product in the catalog.
    """
    print(f">> Fetching price of the product {product} ")
    prices={"laptop":1244.45, "headphones": 540.51, "keyboard":25, "mouse":45.85}
    return prices.get(product, 0)

@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """ Apply a discount tier to the price and return the final price.
        Available tiers: bronze, silver and gold.
    """
    print(f" Applying discount on price:{price} with discount_tier:{discount_tier}")
    discount_percentages={"bronze": 3, "silver": 9, "gold": 27}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount/100), 2)

tools_for_llm=[
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look at the prices of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name",
                        "enum": ["laptop","headphones","mouse","keyboard"]
                    }
                },
                "required": ["product"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier applicable",
                        "enum": ["gold","silver","bronze"]
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]

def call_llm(msgs):
    resp = openai_client.chat.completions.create(
        model=MODEL,
        tools=tools_for_llm,
        messages=msgs
    )
    msgs.append(resp.choices[0].message)
    return resp.choices[0].message  

@traceable(name="Raw Agent Loop without using langchain abstractions")
def run_agent_raw_agent_loop(question: str):
    tools_dict={
        "get_product_price":get_product_price,
        "apply_discount":apply_discount
    }

    print(f"Question: {question}")
    print("="*90)

    messages=[
        {
            "role": "system",
            "content":
                "You are a helpful shopping assistance."
                "You have access to a catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES - You must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. ONLY call apply_discount AFTER you have received "
                "a price from get_product_price. PASS the exact price.\n"
                "returned by get_product_price - do NOT PASS a made-up number.\n"
                "3. ALWAYS use apply_discount tool to calculate discount NEVER use math by yourself for calculating discount.\n"
                "4. If user does not specify discount_tier, "
                "ask them which tier to use - NEVER assume one by yourself."
            
        },
        {
            "role":"user",
            "content":question
        }
    ]

    for iteration in range(1, MAX_ITERATIONS+1):
        print(f"Iteration : ---- {iteration} ----")
        ai_message=call_llm(messages)

        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"{ai_message.content}")
            return ai_message.content

        tool_call=tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        tool_call_id = tool_call.id

        print(f"[Tool Selected]: {tool_name} with arguments {tool_args}")

        tool_to_use = tools_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found.")

        observation = tool_to_use(**tool_args)

        print(f"Tool Result: {observation}")    

        messages.append(ai_message.model_dump())
        messages.append(
            {
                "role": "tool",
                "content": str(observation), 
                "tool_call_id": tool_call_id
            
            }
        )

    print("Error! Maxed out iterations.")
    return None    


