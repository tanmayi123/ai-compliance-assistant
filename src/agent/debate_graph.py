from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage


# ── State definition ───────────────────────────────────────────────────────────
class DebateState(TypedDict):
    question: str
    context: str
    strict_argument: str
    lenient_argument: str
    final_answer: str


# ── LLM ────────────────────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ── Node 1: Strict Agent ───────────────────────────────────────────────────────
def strict_agent(state: DebateState) -> DebateState:
    """Takes the most conservative, risk-averse compliance interpretation."""
    messages = [
        SystemMessage(content="""You are a strict compliance lawyer who always 
gives the most conservative, risk-averse interpretation of regulations. 
You prioritize full compliance over convenience. Always assume the stricter 
reading of any ambiguous rule applies. Be concise."""),
        HumanMessage(content=f"""Context from compliance documents:
{state['context']}

Question: {state['question']}

Give the strictest possible compliance interpretation.""")
    ]
    response = llm.invoke(messages)
    return {**state, "strict_argument": response.content}


# ── Node 2: Lenient Agent ──────────────────────────────────────────────────────
def lenient_agent(state: DebateState) -> DebateState:
    """Takes a more flexible, practical compliance interpretation."""
    messages = [
        SystemMessage(content="""You are a pragmatic compliance advisor who 
gives practical, flexible interpretations of regulations while still staying 
within legal boundaries. You look for reasonable exceptions and practical 
workarounds. Be concise."""),
        HumanMessage(content=f"""Context from compliance documents:
{state['context']}

Question: {state['question']}

Give a practical, flexible compliance interpretation.""")
    ]
    response = llm.invoke(messages)
    return {**state, "lenient_argument": response.content}


# ── Node 3: Synthesizer ────────────────────────────────────────────────────────
def synthesizer(state: DebateState) -> DebateState:
    """Reads both arguments and produces a balanced final answer."""
    messages = [
        SystemMessage(content="""You are a senior compliance expert and mediator. 
You have heard two perspectives on a compliance question — one strict and one lenient. 
Your job is to synthesize both into a balanced, actionable final answer that 
acknowledges the tension, explains both sides, and gives a clear recommendation. 
Always note this is informational, not legal advice."""),
        HumanMessage(content=f"""Question: {state['question']}

Strict interpretation:
{state['strict_argument']}

Lenient interpretation:
{state['lenient_argument']}

Synthesize both perspectives into a balanced final answer with a clear recommendation.""")
    ]
    response = llm.invoke(messages)
    return {**state, "final_answer": response.content}


# ── Build the graph ────────────────────────────────────────────────────────────
def build_debate_graph():
    graph = StateGraph(DebateState)

    graph.add_node("strict_agent", strict_agent)
    graph.add_node("lenient_agent", lenient_agent)
    graph.add_node("synthesizer", synthesizer)

    graph.set_entry_point("strict_agent")
    graph.add_edge("strict_agent", "lenient_agent")
    graph.add_edge("lenient_agent", "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()


# ── Public function to run the debate ─────────────────────────────────────────
def run_debate(question: str, context: str = "") -> dict:
    """
    Run the three-agent debate graph on an ambiguous compliance question.
    Returns a dict with strict_argument, lenient_argument, and final_answer.
    """
    debate = build_debate_graph()
    result = debate.invoke({
        "question": question,
        "context": context,
        "strict_argument": "",
        "lenient_argument": "",
        "final_answer": ""
    })
    return result