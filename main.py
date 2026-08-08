from dotenv import load_dotenv
load_dotenv()   

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama  
from langchain_tavily import TavilySearch
# from tavily import TavilyClient

# tavily = TavilyClient()


# @tool
# def search(query: str) -> str:
#     """
#         Tool that searches over internet
#         Args:
#             query: {weather of the place to search for}
#         Returns:
#             The weather of the place mentioned in query    
#     """
#     print(f"Searching for: {query}")
#     return tavily.search(query=query)

llm=ChatOllama(
    model="llama3.2:latest",
    temperature=0.2
)
tools=[TavilySearch()]

agent101 = create_agent(model=llm, tools=tools)

def main():
    print("Hello, Agent Rover from Main!")
    agent_response = agent101.invoke({"messages":HumanMessage(content="What is the weather of Kolkata today?")})


    
    print(agent_response)


if __name__ == "__main__":
    main()    