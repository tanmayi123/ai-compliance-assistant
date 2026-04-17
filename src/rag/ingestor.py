import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

DOCS_DIR = Path("data/documents")


def get_pinecone_store(embeddings):
    """Initialize and return a PineconeVectorStore."""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME", "compliance-docs")
    index = pc.Index(index_name)
    return PineconeVectorStore(index=index, embedding=embeddings)


def ingest():
    """Load all PDFs from data/documents/, chunk, embed and upsert to Pinecone."""

    pdfs = list(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        print("[ingestor] No PDFs found in data/documents/")
        return

    print(f"[ingestor] Found {len(pdfs)} PDF(s): {[p.name for p in pdfs]}")

    all_docs = []
    for pdf_path in pdfs:
        print(f"[ingestor] Loading {pdf_path.name} ...")
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        all_docs.extend(docs)

    print(f"[ingestor] Total pages loaded: {len(all_docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(all_docs)
    print(f"[ingestor] Total chunks created: {len(chunks)}")

    print("[ingestor] Embedding and upserting to Pinecone ...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = get_pinecone_store(embeddings)
    vectorstore.add_documents(chunks)

    print("[ingestor] Done!")


def ingest_uploaded_file(uploaded_file) -> str:
    """
    Chunk and embed a single uploaded PDF file object (from Streamlit),
    then upsert it into Pinecone.
    Returns a status message.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # ── Save uploaded file to a temp location so PyPDFLoader can read it ──────
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # ── Load and chunk ─────────────────────────────────────────────────────────
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    for doc in docs:
        doc.metadata["source"] = uploaded_file.name

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    # ── Upsert into Pinecone ───────────────────────────────────────────────────
    vectorstore = get_pinecone_store(embeddings)
    vectorstore.add_documents(chunks)

    return f"✅ '{uploaded_file.name}' ingested — {len(chunks)} chunks added to Pinecone."


if __name__ == "__main__":
    ingest()