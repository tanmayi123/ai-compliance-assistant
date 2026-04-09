from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from src.tools.web_search import get_web_search_tool


def build_agent(retriever, memory):
    """
    Build and return a LangChain ReAct agent with two tools:
    1. RAG retriever — searches ingested compliance documents
    2. Tavily web search — for recent updates or out-of-store queries
    """

    # ── Tool 1: RAG retriever ──────────────────────────────────────────────────
    rag_tool = create_retriever_tool(
        retriever,
        name="compliance_document_search",
        description=(
            "Search through official compliance documents including HIPAA, GDPR, "
            "EU AI Act, FINRA rules, and CCPA. Use this first for any question "
            "about these regulations. Input should be the user's question."
        )
    )

    # ── Tool 2: Web search ─────────────────────────────────────────────────────
    web_tool = get_web_search_tool(max_results=3)

    tools = [rag_tool, web_tool]

    # ── LLM ────────────────────────────────────────────────────────────────────
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # ── Prompt — pull standard ReAct prompt from LangChain hub ────────────────
    prompt = hub.pull("hwchase17/react")

    # ── Build ReAct agent ──────────────────────────────────────────────────────
    agent = create_react_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10
    )

    return agent_executor