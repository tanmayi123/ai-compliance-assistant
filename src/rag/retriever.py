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


def get_retriever_with_scores(query: str, k: int = 4):
    """
    Run a similarity search and return (documents, scores).
    Used for citations and confidence meter.
    scores are cosine similarity: 1.0 = perfect match, 0.0 = no match.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME", "compliance-docs")
    index = pc.Index(index_name)
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
    results = vectorstore.similarity_search_with_score(query, k=k)
    # results is a list of (Document, score) tuples
    docs = [r[0] for r in results]
    scores = [r[1] for r in results]
    return docs, scores