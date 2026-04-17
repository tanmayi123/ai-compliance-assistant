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

# ── Map PDF filenames to regulation labels ─────────────────────────────────────
REGULATION_TAGS = {
    "hipaa-simplification-201303.pdf": "hipaa",
    "CELEX_32016R0679_EN_TXT.pdf": "gdpr",
    "ccpa-proposed-regs.pdf": "ccpa",
    "OJ_L_202401689_EN_TXT.pdf": "eu_ai_act",
    "p126234.pdf": "finra",
}

# ── Fallback: detect regulation from filename keywords ────────────────────────
def detect_regulation(filename: str) -> str:
    name = filename.lower()
    if "hipaa" in name:
        return "hipaa"
    elif "gdpr" in name or "celex" in name or "2016r0679" in name:
        return "gdpr"
    elif "ccpa" in name:
        return "ccpa"
    elif "ai_act" in name or "ai-act" in name or "202401689" in name:
        return "eu_ai_act"
    elif "finra" in name or "p126234" in name:
        return "finra"
    return "general"


def get_pinecone_store(embeddings):
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME", "compliance-docs")
    index = pc.Index(index_name)
    return PineconeVectorStore(index=index, embedding=embeddings)


def ingest():
    """Load all PDFs from data/documents/, tag with regulation, chunk, embed and upsert to Pinecone."""
    pdfs = list(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        print("[ingestor] No PDFs found in data/documents/")
        return

    print(f"[ingestor] Found {len(pdfs)} PDF(s): {[p.name for p in pdfs]}")

    all_docs = []
    for pdf_path in pdfs:
        regulation = REGULATION_TAGS.get(pdf_path.name, detect_regulation(pdf_path.name))
        print(f"[ingestor] Loading {pdf_path.name} → regulation: {regulation}")
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        for doc in docs:
            doc.metadata["regulation"] = regulation
            doc.metadata["source"] = pdf_path.name
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
    """Chunk, tag and upsert a single uploaded PDF into Pinecone."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    regulation = REGULATION_TAGS.get(uploaded_file.name, detect_regulation(uploaded_file.name))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = uploaded_file.name
        doc.metadata["regulation"] = regulation

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    vectorstore = get_pinecone_store(embeddings)
    vectorstore.add_documents(chunks)

    return f"✅ '{uploaded_file.name}' ingested as `{regulation}` — {len(chunks)} chunks added to Pinecone."


if __name__ == "__main__":
    ingest()