# ⚖️ AI Compliance Assistant

A production-grade AI-powered compliance assistant built with LangChain, LangGraph, Pinecone, and Streamlit. Ask questions about HIPAA, GDPR, EU AI Act, FINRA, and CCPA — powered by RAG, multi-agent specialist routing, web search, and multi-agent debate.

---

## Features

- **Multi-Agent Specialist Routing** — Supervisor agent classifies each question and routes it to a dedicated specialist (HIPAA, GDPR, EU AI Act, FINRA, CCPA)
- **RAG with Pinecone** — Official compliance documents chunked, embedded, and stored in Pinecone vector database with regulation-level metadata filtering
- **Streaming Responses** — Token-by-token streaming with live status updates (searching docs → searching web → generating)
- **Multi-Agent Debate Mode** — Ambiguous questions trigger a 3-agent LangGraph debate (strict → lenient → synthesizer)
- **Web Search Fallback** — Tavily integration for recent updates not covered in documents
- **Source Citations** — Every answer shows which document and page number it came from
- **Law Updates Dashboard** — Live compliance news fetched via Tavily
- **Compliance Intelligence** — Penalty tracker with Plotly visualizations + compliance calendar
- **PDF Export** — Download any answer as a formatted compliance report
- **ELI5 Mode** — Simplifies answers for non-experts
- **Custom UI** — Dark theme with Cormorant Garamond + DM Sans typography

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | OpenAI GPT-4o-mini |
| Orchestration | LangChain + LangGraph |
| Vector DB | Pinecone |
| Embeddings | OpenAI text-embedding-3-small |
| Web Search | Tavily |
| Frontend | Streamlit |
| PDF Generation | ReportLab |
| Visualizations | Plotly |

---

## Project Structure

```
ai-compliance-assistant/
├── app.py                          # Main Streamlit app
├── requirements.txt
├── .python-version                 # Pins Python 3.11
├── data/
│   └── documents/                  # PDF compliance documents
│       ├── hipaa-simplification-201303.pdf
│       ├── CELEX_32016R0679_EN_TXT.pdf
│       ├── ccpa-proposed-regs.pdf
│       ├── OJ_L_202401689_EN_TXT.pdf
│       └── p126234.pdf
└── src/
    ├── agent/
    │   ├── orchestrator.py         # LangGraph agent builder
    │   ├── supervisor.py           # Multi-agent specialist routing
    │   └── debate_graph.py         # 3-agent debate graph
    ├── rag/
    │   ├── ingestor.py             # PDF ingestion → Pinecone
    │   └── retriever.py            # Pinecone retrieval
    ├── tools/
    │   ├── web_search.py           # Tavily search tool
    │   ├── law_updates.py          # Law updates fetcher
    │   └── compliance_intelligence.py  # Penalties + calendar
    ├── memory/
    │   └── chat_memory.py          # Conversation memory
    └── utils/
        ├── formatter.py            # Response formatting
        ├── style_loader.py         # CSS loader
        ├── styles.css              # Custom UI styles
        └── pdf_exporter.py         # PDF report generation
```

---

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-compliance-assistant.git
cd ai-compliance-assistant
```

### 2. Create and activate a Python 3.11 environment

```bash
conda create -n ai-compliance python=3.11
conda activate ai-compliance
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=compliance-docs
```

### 5. Set up Pinecone

1. Create a free account at [pinecone.io](https://www.pinecone.io)
2. Create an index with:
   - **Name:** `compliance-docs`
   - **Model:** `text-embedding-3-small` (OpenAI)
   - **Dimensions:** `1536`
   - **Metric:** `cosine`

### 6. Ingest compliance documents

Place your PDF documents in `data/documents/` then run:

```bash
python -m src.rag.ingestor
```

This chunks all PDFs, adds regulation metadata tags, and upserts them to Pinecone.

### 7. Run the app

```bash
streamlit run app.py
```

---

## Deploying to Streamlit Cloud

### 1. Push to GitHub

Make sure your repo is on GitHub with all files **except** `.env` (which should be in `.gitignore`).

### 2. Deploy

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **"Create app"**
4. Select your repo, branch (`main`), and main file (`app.py`)
5. Under **Advanced settings** → set **Python version** to `3.11`
6. Click **Deploy**

### 3. Add secrets

Once deployed, go to **Settings → Secrets** and add:

```toml
OPENAI_API_KEY = "sk-..."
TAVILY_API_KEY = "tvly-..."
PINECONE_API_KEY = "..."
PINECONE_INDEX_NAME = "compliance-docs"
```

### 4. Enable login (optional)

Go to **Settings → Sharing** → set to **Private** → add allowed email addresses.

> ⚠️ Note: Streamlit Community Cloud free tier puts apps to sleep after ~7 days of inactivity. The first visitor after sleep will see a "wake up" screen with a ~30 second wait.

---

## Supported Regulations

| Regulation | Coverage |
|-----------|---------|
| 🏥 HIPAA | Privacy Rule, Security Rule, Breach Notification, PHI |
| 🇪🇺 GDPR | Articles 1–99, Data Subject Rights, Controller/Processor obligations |
| 🤖 EU AI Act | Risk classification, Prohibited practices, High-risk AI requirements |
| 💰 FINRA | Rules 2000–9000, Suitability, AML, Supervision |
| 🔒 CCPA/CPRA | Consumer rights, Business obligations, Enforcement |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key (GPT-4o-mini + embeddings) |
| `TAVILY_API_KEY` | Tavily API key for web search |
| `PINECONE_API_KEY` | Pinecone API key |
| `PINECONE_INDEX_NAME` | Pinecone index name (default: `compliance-docs`) |

---

## Disclaimer

This tool provides informational guidance only and does not constitute legal advice. Always consult a qualified legal or compliance professional for guidance specific to your situation.