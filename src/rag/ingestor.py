import tempfile
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

DOCS_DIR = Path("data/documents")
VECTORSTORE_DIR = Path("data/vectorstore")


def ingest():
    """Load all PDFs from data/documents/, chunk, embed and save to FAISS."""

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

    print("[ingestor] Embedding chunks and saving to FAISS ...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))
    print(f"[ingestor] Vectorstore saved to {VECTORSTORE_DIR}")
    print("[ingestor] Done!")


def ingest_uploaded_file(uploaded_file) -> str:
    """
    Chunk and embed a single uploaded PDF file object (from Streamlit),
    then merge it into the existing FAISS vectorstore.
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

    # Tag each chunk with the original filename as source metadata
    for doc in docs:
        doc.metadata["source"] = uploaded_file.name

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    # ── Merge into existing vectorstore or create new one ─────────────────────
    if VECTORSTORE_DIR.exists():
        vectorstore = FAISS.load_local(
            str(VECTORSTORE_DIR),
            embeddings,
            allow_dangerous_deserialization=True
        )
        vectorstore.add_documents(chunks)
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))

    return f"✅ '{uploaded_file.name}' ingested — {len(chunks)} chunks added to the knowledge base."


if __name__ == "__main__":
    ingest()