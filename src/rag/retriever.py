import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()


def get_retriever(k: int = 4):
    """Connect to Pinecone and return a retriever that fetches top-k chunks."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME", "compliance-docs")
    index = pc.Index(index_name)

    vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": k})