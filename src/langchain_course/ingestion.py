import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
# from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pathlib import Path
from .embeddings import load_embedding



load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENT_PATH = PROJECT_ROOT / "medium-blog.txt"




if __name__ == "__main__":
    print("Ingesting...")


loader = TextLoader(
    str(DOCUMENT_PATH),
    encoding="UTF-8"
)
document = loader.load()

print("splitting.....")
text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(document)
print(f"Created ----> {len(texts)} chunks.")

embeddings = load_embedding()

vector = embeddings.embed_query(texts[0].page_content)

print(f"Embedding dimensions: {len(vector)}")
print(f"First 10 values: {vector[:10]}")

### Storing in Pinecone managed Vector DB

vectorstore = PineconeVectorStore.from_documents(
    texts,
    embeddings,
    index_name=os.environ["INDEX_NAME"],
)

print("finish===========")
