import re
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk
from src.utils.formatter import format_response
from src.utils.style_loader import load_styles, welcome_state
from src.utils.pdf_exporter import generate_compliance_pdf
from src.rag.retriever import get_retriever, get_retriever_with_scores
from src.agent.orchestrator import build_agent, should_debate, run_supervisor, SPECIALISTS
from src.rag.ingestor import ingest_uploaded_file
from src.tools.law_updates import fetch_law_updates
from src.tools.compliance_intelligence import fetch_penalty_data, fetch_calendar_data, PENALTY_TOPICS
from src.agent.debate_graph import run_debate

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
load_styles()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style='padding: 24px 20px 16px 20px;'>
    <div style='display:flex; align-items:center; gap:10px; margin-bottom:4px;'>
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#c9a96e" stroke-width="1.5">
            <path d="M12 3L4 7v5c0 4.5 3.3 8.7 8 9.9C17.7 20.7 21 16.5 21 12V7L12 3z"/>
        </svg>
        <span style='font-family: Cormorant Garamond, serif; font-size:1.1rem; color:#c9a96e; letter-spacing:0.03em;'>AI Compliance</span>
    </div>
    <div style='font-size:0.7rem; color:#a8a29e; letter-spacing:0.1em; text-transform:uppercase; padding-left:32px;'>Assistant</div>
</div>
<hr style='border-color:rgba(255,255,255,0.06); margin:0 0 8px 0;'/>
<div style='padding: 8px 20px; font-size:0.68rem; color:#666360; text-transform:uppercase; letter-spacing:0.1em;'>Ask questions about</div>
<div style='padding: 0 16px;'>
    <div style='display:flex; align-items:center; gap:12px; padding:10px 4px; border-bottom:1px solid rgba(255,255,255,0.06); color:#f5f1eb; font-size:0.95rem;'>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#a8a29e" stroke-width="1.5"><path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/></svg>
        HIPAA
    </div>
    <div style='display:flex; align-items:center; gap:12px; padding:10px 4px; border-bottom:1px solid rgba(255,255,255,0.06); color:#f5f1eb; font-size:0.95rem;'>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#a8a29e" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>
        GDPR
    </div>
    <div style='display:flex; align-items:center; gap:12px; padding:10px 4px; border-bottom:1px solid rgba(255,255,255,0.06); color:#f5f1eb; font-size:0.95rem;'>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#a8a29e" stroke-width="1.5"><rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4"/></svg>
        EU AI Act
    </div>
    <div style='display:flex; align-items:center; gap:12px; padding:10px 4px; border-bottom:1px solid rgba(255,255,255,0.06); color:#f5f1eb; font-size:0.95rem;'>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#a8a29e" stroke-width="1.5"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg>
        FINRA
    </div>
    <div style='display:flex; align-items:center; gap:12px; padding:10px 4px; border-bottom:1px solid rgba(255,255,255,0.06); color:#f5f1eb; font-size:0.95rem;'>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#a8a29e" stroke-width="1.5"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>
        CCPA
    </div>
</div>
<hr style='border-color:rgba(255,255,255,0.06); margin:16px 0 8px 0;'/>
""", unsafe_allow_html=True)

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
    st.session_state.lg_messages = []

if "law_updates" not in st.session_state:
    st.session_state.law_updates = []

if "penalties" not in st.session_state:
    st.session_state.penalties = []
    st.session_state.penalties_structured = []

if "calendar" not in st.session_state:
    st.session_state.calendar = []
    st.session_state.calendar_structured = []

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
tab1, tab2, tab3 = st.tabs(["Chat", "Law Updates", "Compliance Intelligence"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.title("AI Compliance Assistant")
    st.caption("Powered by RAG · Web Search · Multi-Agent")

    if not agent_ready:
        st.warning("⚠️ RAG not ready. Please run the ingestor first.")

    if not st.session_state.messages:
        welcome_state()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "risk" in msg:
                st.markdown(f"**Compliance Risk:** {msg['risk']}")
            if msg["role"] == "assistant" and msg.get("specialist"):
                st.caption(f"{msg['specialist']} responded")
            if msg["role"] == "assistant" and msg.get("citations"):
                with st.expander("📄 Sources", expanded=False):
                    seen = set()
                    for c in msg["citations"]:
                        page_str = f"Page {int(c['page']) + 1}" if c["page"] is not None else "Page N/A"
                        key = f"{c['source']}_{page_str}"
                        if key not in seen:
                            seen.add(key)
                            st.markdown(f"• **{c['source']}** — {page_str}")
            if msg["role"] == "assistant" and "risk" in msg:
                msg_idx = st.session_state.messages.index(msg)
                user_q = ""
                if msg_idx > 0:
                    prev = st.session_state.messages[msg_idx - 1]
                    if prev["role"] == "user":
                        user_q = prev["content"]
                pdf_bytes = generate_compliance_pdf(
                    question=user_q,
                    answer=msg["content"],
                    specialist=msg.get("specialist", ""),
                    risk_label=msg.get("risk", ""),
                    citations=msg.get("citations", []),
                )
                st.download_button(
                    label="Download Report",
                    data=pdf_bytes,
                    file_name=f"compliance_report_{msg_idx}.pdf",
                    mime="application/pdf",
                    key=f"dl_{msg_idx}",
                )
            if msg["role"] == "assistant" and msg.get("debate"):
                with st.expander("See both sides of the debate"):
                    st.markdown("**Strict interpretation:**")
                    st.markdown(msg["debate"]["strict"])
                    st.markdown("**Lenient interpretation:**")
                    st.markdown(msg["debate"]["lenient"])

    chat_input_result = st.chat_input(
        "Ask a compliance question...",
        accept_file="multiple",
        file_type=["pdf"],
    )

    if chat_input_result:
        attached_files = chat_input_result.files if hasattr(chat_input_result, "files") else []
        for f in attached_files:
            if f.name not in st.session_state.get("uploaded_files", []):
                with st.spinner(f"Processing {f.name}..."):
                    # Reset file position before reading
                    if hasattr(f, 'seek'):
                        f.seek(0)
                    status = ingest_uploaded_file(f)
                    load_retriever.clear()
                    load_agent.clear()
                    if "uploaded_files" not in st.session_state:
                        st.session_state.uploaded_files = []
                    st.session_state.uploaded_files.append(f.name)
                    st.session_state["upload_status"] = status

    if st.session_state.get("upload_status"):
        st.success(st.session_state["upload_status"])

    if st.session_state.get("uploaded_files"):
        st.caption("Attached: " + " · ".join(st.session_state.uploaded_files))

    user_input = chat_input_result.text if chat_input_result else None

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            debate_data = None

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
                    # ── Fetch citations ────────────────────────────────────
                    citation_docs, _ = get_retriever_with_scores(user_input, k=4)

                    # ── Supervisor: classify + pick specialist ─────────────
                    with st.spinner("🧭 Routing to specialist..."):
                        specialist_key, specialist_graph = run_supervisor(
                            user_input,
                            st.session_state.lg_messages
                        )

                    spec = SPECIALISTS[specialist_key]
                    st.caption(f"{spec['emoji']} **{spec['label']} responding...**")

                    status_placeholder = st.empty()
                    response_placeholder = st.empty()

                    input_messages = list(st.session_state.lg_messages) + [
                        HumanMessage(content=user_input)
                    ]

                    full_response = ""
                    in_final_answer = False
                    buffer = ""
                    BATCH_SIZE = 15

                    for msg, metadata in specialist_graph.stream(
                        {"messages": input_messages, "current_tool": "", "specialist": specialist_key},
                        stream_mode="messages"
                    ):
                        if metadata.get("langgraph_node") == "tools":
                            if not in_final_answer:
                                last = [m for m in input_messages if hasattr(m, "tool_calls")]
                                if last and last[-1].tool_calls:
                                    tname = last[-1].tool_calls[0].get("name", "")
                                    if tname == "compliance_document_search":
                                        status_placeholder.info(f"🔍 Searching {spec['label']} documents...")
                                    elif "tavily" in tname:
                                        status_placeholder.info("🌐 Searching the web...")

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
                            if len(buffer) >= BATCH_SIZE:
                                response_placeholder.markdown(full_response + "▌")
                                buffer = ""

                    status_placeholder.empty()
                    raw_response = full_response
                    response_placeholder.markdown(raw_response)

                    st.session_state.lg_messages = input_messages + [
                        AIMessage(content=raw_response)
                    ]

                    if eli5_mode:
                        display_response = simplify_response(raw_response)
                        st.info("🧒 ELI5 mode is on — answer simplified for clarity.")
                        response_placeholder.markdown(display_response)
                    else:
                        display_response = format_response(raw_response)

                    # ── Sources block ──────────────────────────────────────
                    if citation_docs:
                        with st.expander("📄 Sources", expanded=False):
                            seen = set()
                            for doc in citation_docs:
                                source = doc.metadata.get("source", "Unknown document")
                                page = doc.metadata.get("page", None)
                                page_str = f"Page {int(page) + 1}" if page is not None else "Page N/A"
                                key = f"{source}_{page_str}"
                                if key not in seen:
                                    seen.add(key)
                                    st.markdown(f"• **{source}** — {page_str}")
                else:
                    raw_response = "Agent not ready. Please run the ingestor first."
                    display_response = raw_response
                    st.markdown(display_response)

            if debate_data:
                with st.expander("⚖️ See both sides of the debate"):
                    st.markdown("**🔴 Strict interpretation:**")
                    st.markdown(debate_data["strict"])
                    st.markdown("**🟢 Lenient interpretation:**")
                    st.markdown(debate_data["lenient"])

            risk_label = get_risk_level(raw_response)
            st.markdown(f"**Compliance Risk:** {risk_label}")

            # ── Download button ────────────────────────────────────────────
            if not should_debate(user_input) and agent_ready and raw_response:
                live_citations = [
                    {"source": doc.metadata.get("source", "Unknown"),
                     "page": doc.metadata.get("page", None)}
                    for doc in citation_docs
                ] if citation_docs else []
                pdf_bytes = generate_compliance_pdf(
                    question=user_input,
                    answer=raw_response,
                    specialist=f"{spec['emoji']} {spec['label']}",
                    risk_label=risk_label,
                    citations=live_citations,
                )
                st.download_button(
                    label="Download Report",
                    data=pdf_bytes,
                    file_name=f"compliance_report.pdf",
                    mime="application/pdf",
                    key=f"dl_live_{len(st.session_state.messages)}",
                )

        msg_entry = {
            "role": "assistant",
            "content": display_response,
            "risk": risk_label
        }
        if debate_data:
            msg_entry["debate"] = debate_data
        if not should_debate(user_input) and agent_ready:
            msg_entry["specialist"] = f"{spec['emoji']} {spec['label']}"
            msg_entry["citations"] = [
                {
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", None),
                }
                for doc in citation_docs
            ]
        st.session_state.messages.append(msg_entry)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: LAW UPDATES DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    @st.fragment
    def law_updates_tab():
        st.title("Law Updates Dashboard")
        st.caption("Stay current with the latest compliance and regulatory developments.")

        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("Refresh Updates", use_container_width=True):
                with st.spinner("Searching for latest updates across all regulations..."):
                    st.session_state.law_updates = fetch_law_updates(max_results_per_topic=3)
        with col2:
            if st.session_state.law_updates:
                st.caption(f"Last fetched: {st.session_state.law_updates[0]['fetched_at']}")

        st.divider()

        if not st.session_state.law_updates:
            st.info("Click Refresh Updates to fetch the latest compliance news.")
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
                        st.markdown(f"[Read more]({item['url']})")
                st.divider()

    law_updates_tab()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: COMPLIANCE INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    import plotly.graph_objects as go
    import plotly.express as px
    import pandas as pd

    @st.fragment
    def compliance_intelligence_tab():
        st.title("Compliance Intelligence")
        st.caption("Visual insights into penalties, fines, and upcoming compliance deadlines.")

        # ── Penalty Tracker ────────────────────────────────────────────────
        st.subheader("Penalty & Fine Tracker")

        if st.button("Fetch Latest Penalties", use_container_width=True):
            with st.spinner("Fetching and analyzing recent enforcement actions..."):
                raw, structured = fetch_penalty_data()
                st.session_state.penalties = raw
                st.session_state.penalties_structured = structured

        if not st.session_state.penalties_structured:
            st.info("Click Fetch Latest Penalties to load visualizations.")
        else:
            data = st.session_state.penalties_structured
            df = pd.DataFrame(data)

            if not df.empty and "amount_millions" in df.columns:
                col1, col2 = st.columns(2)

                with col1:
                    reg_totals = df.groupby("regulation")["amount_millions"].sum().reset_index()
                    reg_totals = reg_totals.sort_values("amount_millions", ascending=True)
                    colors = {"HIPAA": "#e74c3c", "GDPR": "#3498db", "EU AI Act": "#9b59b6",
                              "FINRA": "#f39c12", "CCPA": "#2ecc71"}
                    fig_bar = go.Figure(go.Bar(
                        x=reg_totals["amount_millions"],
                        y=reg_totals["regulation"],
                        orientation="h",
                        marker_color=[colors.get(r, "#95a5a6") for r in reg_totals["regulation"]],
                        text=[f"${v:.1f}M" for v in reg_totals["amount_millions"]],
                        textposition="outside"
                    ))
                    fig_bar.update_layout(
                        title="Total Fines by Regulation ($M)",
                        xaxis_title="Amount ($ Millions)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font_color="white",
                        height=300,
                        margin=dict(l=10, r=40, t=40, b=10)
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                with col2:
                    fig_bubble = px.scatter(
                        df[df["amount_millions"] > 0],
                        x="regulation",
                        y="amount_millions",
                        size="amount_millions",
                        color="regulation",
                        hover_name="company",
                        hover_data={"reason": True, "year": True, "amount_millions": ":.1f"},
                        color_discrete_map=colors,
                        title="Fines by Company & Regulation"
                    )
                    fig_bubble.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font_color="white",
                        height=300,
                        showlegend=False,
                        margin=dict(l=10, r=10, t=40, b=10),
                        yaxis_title="Amount ($M)"
                    )
                    st.plotly_chart(fig_bubble, use_container_width=True)

                with st.expander("Full Penalty Data", expanded=False):
                    display_df = df[["regulation", "company", "amount_millions", "reason", "year"]].copy()
                    display_df.columns = ["Regulation", "Company", "Fine ($M)", "Reason", "Year"]
                    display_df = display_df.sort_values("Fine ($M)", ascending=False)
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                    st.markdown("**Sources:**")
                    seen_urls = set()
                    for item in st.session_state.penalties:
                        if item["url"] not in seen_urls and item["url"] != "#":
                            seen_urls.add(item["url"])
                            st.markdown(f"• [{item['title']}]({item['url']})")

        st.divider()

        # ── Compliance Calendar ────────────────────────────────────────────
        st.subheader("Compliance Calendar")

        if st.button("Fetch Upcoming Deadlines", use_container_width=True):
            with st.spinner("Fetching and analyzing upcoming compliance deadlines..."):
                raw_cal, structured_cal = fetch_calendar_data()
                st.session_state.calendar = raw_cal
                st.session_state.calendar_structured = structured_cal

        if not st.session_state.calendar_structured:
            st.info("Click Fetch Upcoming Deadlines to load the compliance calendar.")
        else:
            cal_data = st.session_state.calendar_structured
            cal_df = pd.DataFrame(cal_data)

            if not cal_df.empty:
                colors = {"HIPAA": "#e74c3c", "GDPR": "#3498db", "EU AI Act": "#9b59b6",
                          "FINRA": "#f39c12", "CCPA": "#2ecc71"}
                type_icons = {
                    "Deadline": "●",
                    "Effective Date": "●",
                    "Review": "●",
                    "Enforcement Start": "●"
                }
                type_colors = {
                    "Deadline": "#e74c3c",
                    "Effective Date": "#2ecc71",
                    "Review": "#3498db",
                    "Enforcement Start": "#f39c12"
                }

                reg_sources = {}
                for item in st.session_state.calendar:
                    reg = item["label"]
                    if reg not in reg_sources:
                        reg_sources[reg] = []
                    if item["url"] != "#":
                        reg_sources[reg].append({"title": item["title"], "url": item["url"]})

                for reg in cal_df["regulation"].unique():
                    reg_items = cal_df[cal_df["regulation"] == reg]
                    color = colors.get(reg, "#95a5a6")
                    st.markdown(f"<h4 style='color:{color}; font-family: DM Sans, sans-serif; font-size:1rem; text-transform:uppercase; letter-spacing:0.08em;'>{reg}</h4>", unsafe_allow_html=True)
                    cols = st.columns(min(len(reg_items), 3))
                    for i, (_, row) in enumerate(reg_items.iterrows()):
                        itype = row.get("type", "")
                        dot_color = type_colors.get(itype, "#aaa")
                        with cols[i % 3]:
                            st.markdown(f"""
<div style='background:#1c1c1f; border-left:3px solid {color};
     padding:14px; border-radius:8px; margin-bottom:8px;'>
    <div style='font-size:0.72rem; color:{dot_color}; text-transform:uppercase; letter-spacing:0.06em;'>
        &#9679; {itype}
    </div>
    <div style='font-size:0.88rem; font-weight:500; margin:6px 0; color:#f0ece4; line-height:1.4;'>{row.get('deadline','')}</div>
    <div style='font-size:0.82rem; color:{color};'>&#128197; {row.get('date','TBD')}</div>
</div>""", unsafe_allow_html=True)

                    sources = reg_sources.get(reg, [])
                    if sources:
                        with st.expander(f"Sources for {reg}", expanded=False):
                            for s in sources[:3]:
                                st.markdown(f"• [{s['title']}]({s['url']})")
                    st.markdown("")

                with st.expander("Full Calendar Data", expanded=False):
                    display_cal = cal_df[["regulation", "deadline", "date", "type"]].copy()
                    display_cal.columns = ["Regulation", "Deadline", "Date", "Type"]
                    st.dataframe(display_cal, use_container_width=True, hide_index=True)

    compliance_intelligence_tab()