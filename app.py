import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from src.utils.formatter import format_response
from src.memory.chat_memory import get_memory
from src.rag.retriever import get_retriever
from src.agent.orchestrator import build_agent

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Compliance Assistant", page_icon="⚖️", layout="centered")
st.title("⚖️ AI Compliance Assistant")
st.caption("Ask about HIPAA, GDPR, EU AI Act, financial regulations, and more.")

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = get_memory(k=5)

# ── Load retriever and agent ───────────────────────────────────────────────────
@st.cache_resource
def load_retriever():
    return get_retriever(k=4)

@st.cache_resource
def load_agent(_retriever, _memory):
    return build_agent(_retriever, _memory)

try:
    retriever = load_retriever()
    agent = load_agent(retriever, st.session_state.memory)
    agent_ready = True
except FileNotFoundError as e:
    agent_ready = False
    st.warning(f"⚠️ RAG not ready: {e}")

# ── Display existing chat messages ─────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ─────────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask a compliance question..."):

    # Display user message
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

            formatted = format_response(raw_response)
            st.markdown(formatted)

    st.session_state.messages.append({"role": "assistant", "content": formatted})