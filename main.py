from typing import List

from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


load_dotenv()

class Source(BaseModel):
    """Schema for a source used by the agent."""

    url: str = Field(
        description="The URL of the source"
    )


class AgentResponse(BaseModel):
    """Schema for the agent's final response."""

    answer: str = Field(
        description="The agent's answer to the query"
    )

    sources: List[Source] = Field(
        default_factory=list,
        description="List of sources used to generate the answer"
    )


# --------------------------------------------------
# 4. FREE OpenRouter model
# --------------------------------------------------

llm = ChatOpenAI(
    model="nvidia/nemotron-nano-9b-v2:free",
    temperature=0,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)


# --------------------------------------------------
# 5. Agent
# --------------------------------------------------

tools = [TavilySearch(max_results=2)]

agent101 = create_agent(
    model=llm,
    tools=tools,
    response_format=AgentResponse,
)


# --------------------------------------------------
# 6. Run
# --------------------------------------------------

def main():

    print("Hello, Agent Rover from Main!")

    agent_response = agent101.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Give me one popular job site in India."
                )
            ]
        }
    )

    print("\n===== RAW AGENT RESPONSE =====")
    print(agent_response)

    print("\n===== STRUCTURED RESPONSE =====")
    print(agent_response["structured_response"])


if __name__ == "__main__":
    main()