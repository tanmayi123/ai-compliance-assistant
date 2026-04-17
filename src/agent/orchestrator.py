import operator
from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

# ── Debate triggers ────────────────────────────────────────────────────────────
DEBATE_TRIGGERS = [
    "can i", "can we", "is it allowed", "is it legal", "am i allowed",
    "what if", "what happens if", "penalty", "violation", "fine",
    "without consent", "without permission", "exempt", "exception",
    "loophole", "workaround", "do i need to", "do we need to",
    "is it okay", "is it ok to", "are we required"
]


def should_debate(question: str) -> bool:
    question_lower = question.lower()
    return any(trigger in question_lower for trigger in DEBATE_TRIGGERS)


# ── Agent state ────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_tool: str


# ── System prompt ──────────────────────────────────────────────────────────────
COMPLIANCE_SYSTEM_PROMPT = """You are an AI Compliance Assistant — a formal, \
professional expert in regulatory compliance law. You specialize in the following \
frameworks:

- HIPAA (Health Insurance Portability and Accountability Act)
- GDPR (General Data Protection Regulation)
- EU AI Act
- FINRA (Financial Industry Regulatory Authority) rules
- CCPA (California Consumer Privacy Act)

BEHAVIOR RULES:
1. Always search the compliance document store first before using web search.
2. Provide structured, precise answers. Use bullet points or numbered steps \
where appropriate to improve clarity.
3. Cite the specific regulation, article, or section whenever possible \
(e.g., "Under GDPR Article 17...", "Per HIPAA 45 CFR §164.502...").
4. If a question falls outside the five frameworks above, still answer it to \
the best of your ability, but clearly state at the start: \
"Note: This question falls outside my primary compliance domains (HIPAA, GDPR, \
EU AI Act, FINRA, CCPA). I will do my best to assist, but please consult a \
qualified specialist."
5. Never speculate or fabricate regulatory details. If uncertain, say so and \
recommend consulting a qualified legal or compliance professional.
6. Always end every response with the following disclaimer on its own line:
---
⚠️ This response is for informational purposes only and does not constitute \
legal advice. Please consult a qualified legal or compliance professional for \
guidance specific to your situation."""


def build_agent(retriever, memory=None):
    """
    Build and return a compiled LangGraph agent with full streaming support.
    memory param kept for API compatibility but state is managed via messages.
    """

    # ── Tools ──────────────────────────────────────────────────────────────────
    rag_tool = create_retriever_tool(
        retriever,
        name="compliance_document_search",
        description=(
            "Search through official compliance documents including HIPAA, GDPR, "
            "EU AI Act, FINRA rules, and CCPA. Use this first for any question "
            "about these regulations. Input should be the user's question."
        )
    )

    web_tool = TavilySearchResults(
        max_results=3,
        description=(
            "Use this tool to search the web for recent compliance law updates, "
            "regulations not covered in the document store, or current legal news. "
            "Input should be a clear search query."
        )
    )

    tools = [rag_tool, web_tool]
    tool_node = ToolNode(tools)

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
    llm_with_tools = llm.bind_tools(tools)

    # ── Node: agent ────────────────────────────────────────────────────────────
    def agent_node(state: AgentState):
        system = SystemMessage(content=COMPLIANCE_SYSTEM_PROMPT)
        messages = [system] + list(state["messages"])
        response = llm_with_tools.invoke(messages)
        return {"messages": [response], "current_tool": ""}

    # ── Node: tools with name tracking ────────────────────────────────────────
    def tool_node_with_tracking(state: AgentState):
        last_message = state["messages"][-1]
        tool_name = ""
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            tool_name = last_message.tool_calls[0].get("name", "")
        result = tool_node.invoke(state)
        result["current_tool"] = tool_name
        return result

    # ── Edge: continue to tools or end ────────────────────────────────────────
    def should_continue(state: AgentState):
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    # ── Build graph ────────────────────────────────────────────────────────────
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node_with_tracking)

    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END}
    )
    graph.add_edge("tools", "agent")

    return graph.compile()