import os
from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from langsmith import traceable
from .raw_fnc_calling import run_agent_raw_agent_loop

MAX_ITERATIONS = 5
MODEL = "nvidia/nemotron-nano-9b-v2:free"

@tool
def get_product_price(product: str) -> float:
    """
        Look at the price of the product in the catalog.
    """
    print(f">> Fetching price of the product {product} ")
    prices={"laptop":1244.45, "headphones": 540.51, "keyboard":25, "mouse":45.85}
    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """ Apply a discount tier to the price and return the final price.
        Available tiers: bronze, silver and gold.
    """
    print(f" Applying discount on price:{price} with discount_tier:{discount_tier}")
    discount_percentages={"bronze": 3, "silver": 9, "gold": 27}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount/100), 2)

@traceable(name="LangChain Raw Agent Loop")
def run_agent(question: str):
    tools=[get_product_price, apply_discount]
    tools_dict={t.name: t for t in tools}

    print(f"Tools created: {tools_dict}")

    print(f"Question: {question}")

    llm=init_chat_model(
            f"openai:{MODEL}",
            temperature=0,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
    print("LLM initialized")

    llm_with_tools=llm.bind_tools(tools)
    print("Tools bound")
    print("="*90)

    messages=[
        SystemMessage(
            content=(
                "You are a helpful shopping assistance."
                "You have access to a catalog tool."
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
            ),
        ),
        HumanMessage(content=question)

    ]

    for iteration in range(1, MAX_ITERATIONS+1):
        print(f"Iteration : ---- {iteration} ----")
        ai_message=llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"{ai_message.content}")
            return ai_message.content


        # Process only the FIRST Toll Call: Force one tool per iteration
        tool_call=tool_calls[0]
        tool_name=tool_call.get("name")
        tool_args=tool_call.get("args", {})
        tool_call_id=tool_call.get("id")

        print(f"[Tool Selected]: {tool_name} with arguments {tool_args}")

        tool_to_use = tools_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found.")

        observation = tool_to_use.invoke(tool_args)

        print(f"Tool Result: {observation}")    

        messages.append(ai_message)
        messages.append(ToolMessage(
                content=str(observation), tool_call_id=tool_call_id
            )
        )

    print("Error! Maxed out iterations.")
    return None    



def main():
    #========== LanngChain Agent using tools
    # print("Hello World here here look at the screen we are going to bind tools to our langchain agent!!!!")
    # print()
    # result=run_agent("What is the price of a laptop after applying a gold discount?")

    # Raw Function calling using provider sdks
    print("Hi Rover! Let's see what raw meat we are eating today Ha Ha Ha...")
    run_agent_raw_agent_loop(
        "What is the price of a headphones with a gold discount?"
    )