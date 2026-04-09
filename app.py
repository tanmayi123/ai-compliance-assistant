import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from src.utils.formatter import format_response
from src.memory.chat_memory import get_memory

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Compliance Assistant", page_icon="⚖️", layout="centered")
st.title("⚖️ AI Compliance Assistant")
st.caption("Ask me anything about HIPAA, GDPR, EU AI Act, financial regulations, and more.")

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert AI compliance and regulatory assistant.

You help users understand complex frameworks such as HIPAA, GDPR, the EU AI Act, 
and financial regulations in a clear and practical way.

Guidelines:
- Provide clear, structured, and easy-to-understand answers
- Break down complex regulations into simple explanations when needed
- Highlight key requirements, risks, and implications
- Use examples where helpful

Safety:
- Your responses are for informational purposes only, not legal advice
- Do not make assumptions or fabricate details
- If unsure or information is incomplete, clearly state it

Tone:
- Be concise, professional, and helpful
"""

# ── Session state: initialize chat history and memory ─────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

if "memory" not in st.session_state:
    st.session_state.memory = get_memory()

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── Display existing chat messages (skip system message) ──────────────────────
for msg in st.session_state.messages[1:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask a compliance question..."):

    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build message list for LLM
    lc_messages = []
    for msg in st.session_state.messages:
        if msg["role"] == "system":
            lc_messages.append(SystemMessage(content=msg["content"]))
        elif msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))

    # Call LLM and display response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = llm.invoke(lc_messages)
            formatted = format_response(response.content)
            st.markdown(formatted)

    # Save assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": formatted})