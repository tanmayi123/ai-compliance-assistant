import re
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from src.utils.formatter import format_response
from src.rag.retriever import get_retriever
from src.agent.orchestrator import build_agent, should_debate
from src.rag.ingestor import ingest_uploaded_file
from src.tools.law_updates import fetch_law_updates
from src.agent.debate_graph import run_debate
from langchain_core.messages import HumanMessage

load_dotenv()

# ── Cache functions ────────────────────────────────────────────────────────────
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

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.lg_messages = []
        st.rerun()

    st.caption("⚠️ This tool provides informational guidance only, not legal advice.")

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "lg_messages" not in st.session_state:
    st.session_state.lg_messages = []  # LangGraph message history for context

if "law_updates" not in st.session_state:
    st.session_state.law_updates = []

# ── Load agent ─────────────────────────────────────────────────────────────────
try:
    retriever = load_retriever()
    agent = load_agent(retriever, None)
    agent_ready = True
except Exception as e:
    agent_ready = False

# ── Helpers ────────────────────────────────────────────────────────────────────
def get_risk_level(response_text: str) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
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
    st.caption("Powered by RAG + Web Search + Multi-Agent Debate")

    if not agent_ready:
        st.warning("⚠️ RAG not ready. Please run the ingestor first.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "risk" in msg:
                st.markdown(f"**Compliance Risk:** {msg['risk']}")
            if msg["role"] == "assistant" and msg.get("debate"):
                with st.expander("⚖️ See both sides of the debate"):
                    st.markdown("**🔴 Strict interpretation:**")
                    st.markdown(msg["debate"]["strict"])
                    st.markdown("**🟢 Lenient interpretation:**")
                    st.markdown(msg["debate"]["lenient"])

    # ── Chat input with native file attachment (Streamlit 1.43+) ──────────────
    chat_input_result = st.chat_input(
        "Ask a compliance question...",
        accept_file="multiple",
        file_type=["pdf"],
    )

    if chat_input_result:
        user_input = chat_input_result.text
        attached_files = chat_input_result.files if hasattr(chat_input_result, "files") else []

        # Process any attached PDFs
        for f in attached_files:
            if f.name not in st.session_state.get("uploaded_files", []):
                with st.spinner(f"Processing {f.name}..."):
                    status = ingest_uploaded_file(f)
                    load_retriever.clear()
                    load_agent.clear()
                    if "uploaded_files" not in st.session_state:
                        st.session_state.uploaded_files = []
                    st.session_state.uploaded_files.append(f.name)
                    st.success(status)

    if st.session_state.get("uploaded_files"):
        st.caption("📎 Added: " + " · ".join(st.session_state.uploaded_files))

    if chat_input_result and chat_input_result.text:
        user_input = chat_input_result.text
    else:
        user_input = None

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            debate_data = None

            # ── Route: debate or normal agent ─────────────────────────────
            if should_debate(user_input):
                with st.spinner("⚖️ Debate mode triggered — consulting multiple perspectives..."):
                    context = ""
                    if agent_ready:
                        docs = retriever.invoke(user_input)
                        for doc in docs:
                            context += doc.page_content + "\n\n"
                    debate_result = run_debate(user_input, context)
                    raw_response = debate_result["final_answer"]
                    debate_data = {
                        "strict": debate_result["strict_argument"],
                        "lenient": debate_result["lenient_argument"]
                    }
                display_response = format_response(raw_response)
                st.markdown(display_response)

            else:
                if agent_ready:
                    # ── Status placeholders ────────────────────────────────
                    status_placeholder = st.empty()
                    response_placeholder = st.empty()
                    raw_response = ""

                    # ── Stream through LangGraph ───────────────────────────
                    from langchain_core.messages import HumanMessage as LGHumanMessage, AIMessageChunk
                    input_messages = list(st.session_state.lg_messages) + [
                        LGHumanMessage(content=user_input)
                    ]

                    full_response = ""
                    in_final_answer = False
                    buffer = ""
                    BATCH_SIZE = 15  # render every N tokens for smoothness

                    for msg, metadata in agent.stream(
                        {"messages": input_messages, "current_tool": ""},
                        stream_mode="messages"
                    ):
                        # ── Tool call status updates ───────────────────────
                        if metadata.get("langgraph_node") == "tools":
                            if not in_final_answer:
                                last = [m for m in input_messages if hasattr(m, "tool_calls")]
                                if last and last[-1].tool_calls:
                                    tname = last[-1].tool_calls[0].get("name", "")
                                    if tname == "compliance_document_search":
                                        status_placeholder.info("🔍 Searching compliance documents...")
                                    elif "tavily" in tname:
                                        status_placeholder.info("🌐 Searching the web...")

                        # ── Stream tokens from final agent response ────────
                        if (
                            metadata.get("langgraph_node") == "agent"
                            and isinstance(msg, AIMessageChunk)
                            and msg.content
                            and not getattr(msg, "tool_calls", None)
                        ):
                            if not in_final_answer:
                                in_final_answer = True
                                status_placeholder.info("✍️ Generating answer...")

                            buffer += msg.content
                            full_response += msg.content

                            # Only re-render every BATCH_SIZE tokens
                            if len(buffer) >= BATCH_SIZE:
                                response_placeholder.markdown(full_response + "▌")
                                buffer = ""

                    # Final render without cursor
                    status_placeholder.empty()
                    raw_response = full_response
                    response_placeholder.markdown(raw_response)

                    # ── Update LangGraph message history ───────────────────
                    from langchain_core.messages import AIMessage
                    st.session_state.lg_messages = input_messages + [
                        AIMessage(content=raw_response)
                    ]

                    if eli5_mode:
                        display_response = simplify_response(raw_response)
                        st.info("🧒 ELI5 mode is on — answer simplified for clarity.")
                        response_placeholder.markdown(display_response)
                    else:
                        display_response = format_response(raw_response)
                else:
                    raw_response = "Agent not ready. Please run the ingestor first."
                    display_response = raw_response
                    st.markdown(display_response)

            # ── Show debate sides if applicable ───────────────────────────
            if debate_data:
                with st.expander("⚖️ See both sides of the debate"):
                    st.markdown("**🔴 Strict interpretation:**")
                    st.markdown(debate_data["strict"])
                    st.markdown("**🟢 Lenient interpretation:**")
                    st.markdown(debate_data["lenient"])

            risk_label = get_risk_level(raw_response)
            st.markdown(f"**Compliance Risk:** {risk_label}")

        msg_entry = {
            "role": "assistant",
            "content": display_response,
            "risk": risk_label
        }
        if debate_data:
            msg_entry["debate"] = debate_data
        st.session_state.messages.append(msg_entry)
        st.rerun()

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