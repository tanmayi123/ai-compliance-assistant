import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from src.utils.formatter import format_response
from src.memory.chat_memory import get_memory
from src.rag.retriever import get_retriever
from src.agent.orchestrator import build_agent

load_dotenv()

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

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.memory = get_memory(k=5)
        st.rerun()

    st.caption("⚠️ This tool provides informational guidance only, not legal advice.")

# ── Main title ─────────────────────────────────────────────────────────────────
st.title("⚖️ AI Compliance Assistant")
st.caption("Powered by RAG + Web Search · Sources cited from official documents")

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

# ── Helper: classify risk level ────────────────────────────────────────────────
def get_risk_level(response_text: str) -> tuple[str, str]:
    """Ask the LLM to classify the compliance risk level of the response."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    classification_prompt = f"""Based on the following compliance answer, classify the risk level as one of:
- LOW: General information, no immediate action needed
- MEDIUM: Some compliance considerations, review recommended  
- HIGH: Serious compliance issue, legal consultation strongly advised

Answer with only one word: LOW, MEDIUM, or HIGH.

Response to classify:
{response_text}
"""
    result = llm.invoke([HumanMessage(content=classification_prompt)])
    level = result.content.strip().upper()

    if "HIGH" in level:
        return "🔴 High Risk", "red"
    elif "MEDIUM" in level:
        return "🟡 Medium Risk", "orange"
    else:
        return "🟢 Low Risk", "green"


# ── Helper: ELI5 rewrite ───────────────────────────────────────────────────────
def simplify_response(response_text: str) -> str:
    """Rewrite a compliance answer in simple plain language."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    eli5_prompt = f"""Rewrite the following compliance answer so that a non-expert 
person with no legal background can understand it easily. Use simple words, 
short sentences, and everyday examples where helpful. Keep it accurate but friendly.

Original answer:
{response_text}
"""
    result = llm.invoke([HumanMessage(content=eli5_prompt)])
    return result.content.strip()


# ── Display existing chat messages ─────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "risk" in msg:
            st.markdown(f"**Compliance Risk:** {msg['risk']}")

# ── Chat input ─────────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask a compliance question..."):

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and thinking..."):

            # ── Get agent response ─────────────────────────────────────────────
            if agent_ready:
                result = agent.invoke({"input": user_input})
                raw_response = result.get("output", "No response received.")
            else:
                raw_response = "Agent not ready. Please run the ingestor first."

            # ── ELI5 rewrite if toggled ────────────────────────────────────────
            if eli5_mode:
                display_response = simplify_response(raw_response)
                st.info("🧒 ELI5 mode is on — answer simplified for clarity.")
            else:
                display_response = format_response(raw_response)

            st.markdown(display_response)

            # ── Risk level indicator ───────────────────────────────────────────
            risk_label, risk_color = get_risk_level(raw_response)
            st.markdown(f"**Compliance Risk:** {risk_label}")

    # Save to session
    st.session_state.messages.append({
        "role": "assistant",
        "content": display_response,
        "risk": risk_label
    })