import os
from langchain_openai import ChatOpenAI


def load_llm():
    return ChatOpenAI(
        model=os.environ["LLM_MODEL"],
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=os.environ["OPENROUTER_BASE_URL"],
        temperature=0
    )