import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from src.utils.formatter import format_response
from src.memory.chat_memory import get_memory
from src.rag.retriever import get_retriever

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Compliance Assistant", page_icon="⚖️", layout="centered")
st.title("⚖️ AI Compliance Assistant")
st.caption("Ask about HIPAA, GDPR, EU AI Act, financial regulations, and more.")

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert AI compliance and legal assistant.
You help users understand complex regulations such as HIPAA, GDPR, EU AI Act,
and financial compliance rules.

Guidelines:
- Answer ONLY based on the context provided from official documents
- Always clarify that your answers are informational, not legal advice
- If the context does not contain the answer, say so clearly
- Be concise but thorough
- Always mention which regulation or document your answer comes from
- Use the conversation history to understand follow-up questions
"""

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = get_memory(k=5)

# ── Load LLM and retriever ─────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

@st.cache_resource
def load_retriever():
    return get_retriever(k=4)

try:
    retriever = load_retriever()
    rag_ready = True
except FileNotFoundError as e:
    rag_ready = False
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

    # ── Retrieve relevant chunks from documents ────────────────────────────────
    context = ""
    sources = []
    if rag_ready:
        retrieved_docs = retriever.invoke(user_input)
        for doc in retrieved_docs:
            context += doc.page_content + "\n\n"
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            sources.append(f"{Path(source).name} (page {page})")

    # ── Build message list: system + memory history + current question ─────────
    lc_messages = [SystemMessage(content=SYSTEM_PROMPT)]

    # Inject memory — last k exchanges
    memory_messages = st.session_state.memory.chat_memory.messages
    for mem_msg in memory_messages:
        lc_messages.append(mem_msg)

    # Build context-enriched user message
    if context:
        context_prompt = f"""Use the following excerpts from official compliance documents to answer the question.
Consider the conversation history above for any follow-up context.

CONTEXT:
{context}

QUESTION: {user_input}
"""
    else:
        context_prompt = user_input

    lc_messages.append(HumanMessage(content=context_prompt))

    # ── Call LLM ───────────────────────────────────────────────────────────────
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and thinking..."):
            response = llm.invoke(lc_messages)
            formatted = format_response(response.content)
            st.markdown(formatted)

            if sources:
                unique_sources = list(dict.fromkeys(sources))
                st.markdown("---")
                st.caption("📄 **Sources:** " + " · ".join(unique_sources))

    # ── Save to session state and memory ──────────────────────────────────────
    st.session_state.messages.append({"role": "assistant", "content": formatted})
    st.session_state.memory.chat_memory.add_user_message(user_input)
    st.session_state.memory.chat_memory.add_ai_message(formatted)