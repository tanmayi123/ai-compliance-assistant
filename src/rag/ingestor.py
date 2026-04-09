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
    """Load PDFs, chunk them, embed and save to FAISS vectorstore."""

    # ── Step 1: Load all PDFs ─────────────────────────────────────────────────
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

    # ── Step 2: Chunk the documents ───────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(all_docs)
    print(f"[ingestor] Total chunks created: {len(chunks)}")

    # ── Step 3: Embed and save to FAISS ───────────────────────────────────────
    print("[ingestor] Embedding chunks and saving to FAISS ...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))
    print(f"[ingestor] Vectorstore saved to {VECTORSTORE_DIR}")
    print("[ingestor] Done! You can now run the app.")


if __name__ == "__main__":
    ingest()