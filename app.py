import streamlit as st
import re
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from src.utils.formatter import format_response
from src.memory.chat_memory import get_memory
from src.rag.retriever import get_retriever
from src.agent.orchestrator import build_agent
from src.rag.ingestor import ingest_uploaded_file
from src.tools.law_updates import fetch_law_updates, UPDATE_TOPICS

load_dotenv()

# ── Cache functions defined first ──────────────────────────────────────────────
@st.cache_resource
def load_retriever():
    return get_retriever(k=4)

@st.cache_resource
def load_agent(_retriever, _memory):
    return build_agent(_retriever, _memory)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Compliance Assistant", page_icon="⚖️", layout="centered")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚖️ AI Compliance Assistant")
    st.markdown("Ask questions about:")
    st.markdown("- 🏥 HIPAA\n- 🇪🇺 GDPR\n- 🤖 EU AI Act\n- 💰 FINRA\n- 🔒 CCPA")
    st.divider()

    eli5_mode = st.toggle(
        "🧒 Explain Like I'm 5 (ELI5)",
        value=False,
        help="Rewrites the answer in simple, plain language"
    )
    st.divider()

    # ── Document uploader ──────────────────────────────────────────────────────
    st.subheader("📂 Upload Your Document")
    st.caption("Upload a PDF to add it to the knowledge base instantly.")

    uploaded_file = st.file_uploader(
        "Drop a PDF here",
        type=["pdf"],
        help="Your document will be chunked, embedded, and searchable immediately."
    )

    if uploaded_file is not None:
        if uploaded_file.name not in st.session_state.get("uploaded_files", []):
            with st.spinner(f"Processing {uploaded_file.name}..."):
                status = ingest_uploaded_file(uploaded_file)
                load_retriever.clear()
                load_agent.clear()
                if "uploaded_files" not in st.session_state:
                    st.session_state.uploaded_files = []
                st.session_state.uploaded_files.append(uploaded_file.name)
                st.success(status)
        else:
            st.info(f"'{uploaded_file.name}' already in knowledge base.")

    if st.session_state.get("uploaded_files"):
        st.caption("**Added this session:**")
        for fname in st.session_state.uploaded_files:
            st.caption(f"• {fname}")

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.memory = get_memory(k=5)
        st.rerun()

    st.caption("⚠️ This tool provides informational guidance only, not legal advice.")

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = get_memory(k=5)

if "law_updates" not in st.session_state:
    st.session_state.law_updates = []

# ── Load agent ─────────────────────────────────────────────────────────────────
try:
    retriever = load_retriever()
    agent = load_agent(retriever, st.session_state.memory)
    agent_ready = True
except FileNotFoundError as e:
    agent_ready = False

# ── Helpers ────────────────────────────────────────────────────────────────────
def get_risk_level(response_text: str) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"""Classify the compliance risk level of this answer as LOW, MEDIUM, or HIGH.
Answer with only one word.

Response: {response_text}
"""
    result = llm.invoke([HumanMessage(content=prompt)])
    level = result.content.strip().upper()
    if "HIGH" in level:
        return "🔴 High Risk"
    elif "MEDIUM" in level:
        return "🟡 Medium Risk"
    return "🟢 Low Risk"


def simplify_response(response_text: str) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"""Rewrite this compliance answer in simple plain language for a non-expert.

Original: {response_text}
"""
    result = llm.invoke([HumanMessage(content=prompt)])
    return result.content.strip()


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["💬 Chat", "📰 Law Updates"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.title("⚖️ AI Compliance Assistant")
    st.caption("Powered by RAG + Web Search · Sources cited from official documents")

    if not agent_ready:
        st.warning("⚠️ RAG not ready. Please run the ingestor first.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "risk" in msg:
                st.markdown(f"**Compliance Risk:** {msg['risk']}")

    if user_input := st.chat_input("Ask a compliance question..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if agent_ready:
                    result = agent.invoke({"input": user_input})
                    raw_response = result.get("output", "No response received.")
                else:
                    raw_response = "Agent not ready. Please run the ingestor first."

                if eli5_mode:
                    display_response = simplify_response(raw_response)
                    st.info("🧒 ELI5 mode is on — answer simplified for clarity.")
                else:
                    display_response = format_response(raw_response)

                st.markdown(display_response)
                risk_label = get_risk_level(raw_response)
                st.markdown(f"**Compliance Risk:** {risk_label}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": display_response,
            "risk": risk_label
        })

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: LAW UPDATES DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.title("📰 Law Updates Dashboard")
    st.caption("Stay current with the latest compliance and regulatory developments.")

    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("🔄 Refresh Updates", use_container_width=True):
            with st.spinner("Searching for latest updates across all regulations..."):
                st.session_state.law_updates = fetch_law_updates(max_results_per_topic=3)

    with col2:
        if st.session_state.law_updates:
            st.caption(f"Last fetched: {st.session_state.law_updates[0]['fetched_at']}")

    st.divider()

    if not st.session_state.law_updates:
        st.info("Click **Refresh Updates** to fetch the latest compliance news.")
    else:
        # Group results by regulation label
        grouped = {}
        for item in st.session_state.law_updates:
            label = item["label"]
            if label not in grouped:
                grouped[label] = []
            grouped[label].append(item)

        for label, items in grouped.items():
            st.subheader(label)
            for item in items:
                with st.expander(item["title"]):
                    clean_content = re.sub(r'\*\*|__|\*|_|#{1,6}\s', '', item["content"])
                    st.markdown(clean_content)
                    st.markdown(f"[🔗 Read more]({item['url']})")
            st.divider()

