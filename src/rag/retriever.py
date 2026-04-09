from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

VECTORSTORE_DIR = Path("data/vectorstore")


def get_retriever(k: int = 4):
    """Load FAISS vectorstore and return a retriever that fetches top-k chunks."""
    if not VECTORSTORE_DIR.exists():
        raise FileNotFoundError(
            "Vectorstore not found. Please run the ingestor first:\n"
            "  python -m src.rag.ingestor"
        )

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})