import os
import operator
from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, AIMessageChunk
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools.retriever import create_retriever_tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from pinecone import Pinecone

load_dotenv()


# ── Agent state ────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_tool: str
    specialist: str  # which specialist was chosen


# ── Specialist definitions ─────────────────────────────────────────────────────
SPECIALISTS = {
    "hipaa": {
        "emoji": "🏥",
        "label": "HIPAA Specialist",
        "regulation": "hipaa",
        "prompt": """You are a HIPAA Compliance Specialist — a formal expert in the \
Health Insurance Portability and Accountability Act.

Your expertise covers:
- Protected Health Information (PHI) and ePHI
- Privacy Rule (45 CFR Part 164, Subpart E)
- Security Rule (45 CFR Part 164, Subpart C)
- Breach Notification Rule
- HIPAA covered entities and business associates
- Minimum necessary standard
- Patient rights under HIPAA

Always cite specific CFR sections (e.g. "Per 45 CFR §164.502...").
Always end with: ⚠️ This is informational only, not legal advice."""
    },
    "gdpr": {
        "emoji": "🇪🇺",
        "label": "GDPR Specialist",
        "regulation": "gdpr",
        "prompt": """You are a GDPR Compliance Specialist — a formal expert in the \
General Data Protection Regulation (EU) 2016/679.

Your expertise covers:
- Lawful bases for processing (Article 6)
- Data subject rights (Articles 15–22)
- Controller and processor obligations
- Data Protection Officer requirements
- Cross-border data transfers
- Privacy by design and by default
- GDPR fines and enforcement

Always cite specific GDPR Articles (e.g. "Under GDPR Article 17...").
Always end with: ⚠️ This is informational only, not legal advice."""
    },
    "eu_ai_act": {
        "emoji": "🤖",
        "label": "EU AI Act Specialist",
        "regulation": "eu_ai_act",
        "prompt": """You are an EU AI Act Compliance Specialist — a formal expert in \
the EU Artificial Intelligence Act (Regulation 2024/1689).

Your expertise covers:
- AI system risk classification (unacceptable, high, limited, minimal)
- Prohibited AI practices
- High-risk AI system requirements
- Conformity assessments and CE marking
- Transparency obligations
- Foundation model and GPAI requirements
- Enforcement and penalties

Always cite specific EU AI Act Articles.
Always end with: ⚠️ This is informational only, not legal advice."""
    },
    "finra": {
        "emoji": "💰",
        "label": "FINRA Specialist",
        "regulation": "finra",
        "prompt": """You are a FINRA Compliance Specialist — a formal expert in \
Financial Industry Regulatory Authority rules and securities regulations.

Your expertise covers:
- FINRA rulebook (FINRA Rules 2000–9000)
- Suitability and best interest standards
- Anti-money laundering (AML) requirements
- Books and records obligations
- Supervision requirements
- Communications with the public
- Registration and licensing

Always cite specific FINRA Rule numbers (e.g. "Under FINRA Rule 2111...").
Always end with: ⚠️ This is informational only, not legal advice."""
    },
    "ccpa": {
        "emoji": "🔒",
        "label": "CCPA Specialist",
        "regulation": "ccpa",
        "prompt": """You are a CCPA Compliance Specialist — a formal expert in the \
California Consumer Privacy Act and its amendments (CPRA).

Your expertise covers:
- Consumer rights (know, delete, opt-out, correct, limit)
- Business obligations and thresholds
- Categories of personal information
- Sensitive personal information
- Service provider and contractor requirements
- CCPA enforcement and penalties
- CPRA amendments

Always cite specific CCPA sections (e.g. "Under Cal. Civ. Code §1798.100...").
Always end with: ⚠️ This is informational only, not legal advice."""
    },
    "general": {
        "emoji": "⚖️",
        "label": "General Compliance Specialist",
        "regulation": None,  # searches all docs
        "prompt": """You are an AI Compliance Assistant — a formal expert in \
regulatory compliance law covering HIPAA, GDPR, EU AI Act, FINRA, and CCPA.

Answer the question using all available compliance knowledge.
Always end with: ⚠️ This is informational only, not legal advice."""
    }
}


# ── Supervisor: classifies question → picks specialist ────────────────────────
def classify_question(question: str) -> str:
    """Use LLM to classify which regulation the question is about."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"""Classify this compliance question into exactly one category.
Reply with only one word from this list: hipaa, gdpr, eu_ai_act, finra, ccpa, general

Question: {question}

Rules:
- hipaa: questions about health data, PHI, medical privacy, healthcare compliance
- gdpr: questions about EU data protection, GDPR articles, EU privacy law
- eu_ai_act: questions about AI regulation, AI risk classification, EU AI rules
- finra: questions about financial industry rules, securities, broker-dealers
- ccpa: questions about California privacy law, CCPA/CPRA consumer rights
- general: questions spanning multiple regulations or unclear domain

Reply with one word only:"""

    result = llm.invoke([HumanMessage(content=prompt)])
    classification = result.content.strip().lower()
    if classification not in SPECIALISTS:
        classification = "general"
    return classification


# ── Build a filtered retriever for a specific regulation ──────────────────────
def get_filtered_retriever(regulation: str | None, k: int = 4):
    """Return a Pinecone retriever filtered to a specific regulation."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME", "compliance-docs")
    index = pc.Index(index_name)
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings)

    if regulation:
        return vectorstore.as_retriever(
            search_kwargs={
                "k": k,
                "filter": {"regulation": {"$eq": regulation}}
            }
        )
    return vectorstore.as_retriever(search_kwargs={"k": k})


# ── Build specialist agent graph ───────────────────────────────────────────────
def build_specialist_graph(specialist_key: str):
    """Build a LangGraph agent for a specific specialist."""
    spec = SPECIALISTS[specialist_key]
    retriever = get_filtered_retriever(spec["regulation"])

    rag_tool = create_retriever_tool(
        retriever,
        name="compliance_document_search",
        description=f"Search {spec['label']} compliance documents. Use this first."
    )

    web_tool = TavilySearchResults(
        max_results=3,
        description="Search the web for recent compliance updates or news."
    )

    tools = [rag_tool, web_tool]
    tool_node = ToolNode(tools)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState):
        system = SystemMessage(content=spec["prompt"])
        messages = [system] + list(state["messages"])
        response = llm_with_tools.invoke(messages)
        return {"messages": [response], "current_tool": "", "specialist": specialist_key}

    def tool_node_with_tracking(state: AgentState):
        last_message = state["messages"][-1]
        tool_name = ""
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            tool_name = last_message.tool_calls[0].get("name", "")
        result = tool_node.invoke(state)
        result["current_tool"] = tool_name
        result["specialist"] = specialist_key
        return result

    def should_continue(state: AgentState):
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node_with_tracking)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


# ── Main entry: classify + run specialist ──────────────────────────────────────
def run_supervisor(question: str, lg_messages: list) -> tuple:
    """
    Classify the question, pick the right specialist, return
    (specialist_key, compiled_graph) so app.py can stream it.
    """
    specialist_key = classify_question(question)
    graph = build_specialist_graph(specialist_key)
    return specialist_key, graph