from dotenv import load_dotenv  # pyright: ignore[reportMissingImports]
load_dotenv()   

from langchain.agents import create_agent  # pyright: ignore[reportMissingImports]
from langchain.tools import tool  # pyright: ignore[reportMissingImports]
from langchain_core.messages import HumanMessage  # pyright: ignore[reportMissingImports]
from langchain_ollama import ChatOllama  # pyright: ignore[reportMissingImports]

@tool  # pyright: ignore[reportUndefinedVariable]
def search(query: str) -> str:
    """
        Tool that searches over internet
        Args:
            query: the place to search for
        Returns:
            The waether of the place mentioned in query    
    """
    print(f"Searching for: {query}")
    return "Kolkata is having heavy monsoon rain "

llm=ChatOllama(
    model="llama3.2:latest",
    temperature=0.2
)
tools=[search]

agent101 = create_agent(model=llm, tools=tools)

def main():
    print("Hello, Agent Rover from Main!")
    agent_response = agent101.invoke({"messages":HumanMessage(content="What is the weather of Kolkata?")})


    
    print(agent_response)


if __name__ == "__main__":
    main()    