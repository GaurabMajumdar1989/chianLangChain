import os

from langchain_core.embeddings import Embeddings
from openai import OpenAI


class OpenRouterEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(
            base_url=os.environ["OPENROUTER_BASE_URL"],
            api_key=api_key,
        )
        self.model = model

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
            encoding_format="float",
        )
        return response.data[0].embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            encoding_format="float",
        )
        return [item.embedding for item in response.data]


def load_embedding():
    return OpenRouterEmbeddings(
        api_key=os.environ["OPENROUTER_API_KEY"],
        model=os.environ["EMBEDDING_MODEL"],
    )