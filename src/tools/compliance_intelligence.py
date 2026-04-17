import os
import re
from datetime import datetime
from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import json

# ── Penalty tracker queries ────────────────────────────────────────────────────
PENALTY_TOPICS = [
    {"label": "HIPAA", "emoji": "🏥", "color": "#e74c3c", "query": "HIPAA violation fines penalties enforcement actions 2024 2025"},
    {"label": "GDPR", "emoji": "🇪🇺", "color": "#3498db", "query": "GDPR fines penalties enforcement decisions 2024 2025"},
    {"label": "EU AI Act", "emoji": "🤖", "color": "#9b59b6", "query": "EU AI Act enforcement penalties violations 2024 2025"},
    {"label": "FINRA", "emoji": "💰", "color": "#f39c12", "query": "FINRA fines enforcement actions sanctions 2024 2025"},
    {"label": "CCPA", "emoji": "🔒", "color": "#2ecc71", "query": "CCPA CPRA fines enforcement penalties 2024 2025"},
]

CALENDAR_TOPICS = [
    {"label": "HIPAA", "emoji": "🏥", "query": "HIPAA compliance deadlines upcoming requirements 2025 2026"},
    {"label": "GDPR", "emoji": "🇪🇺", "query": "GDPR compliance deadlines upcoming enforcement dates 2025 2026"},
    {"label": "EU AI Act", "emoji": "🤖", "query": "EU AI Act implementation deadlines compliance dates 2025 2026"},
    {"label": "FINRA", "emoji": "💰", "query": "FINRA rule compliance deadlines effective dates 2025 2026"},
    {"label": "CCPA", "emoji": "🔒", "query": "CCPA CPRA compliance deadlines requirements 2025 2026"},
]


def extract_penalty_data(raw_results: list) -> list:
    """Use LLM to extract structured fine data from raw Tavily results."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    text = "\n\n".join([f"{r['label']}: {r['title']}\n{r['content']}" for r in raw_results])

    prompt = f"""Extract fine/penalty data from these compliance news results.
Return a JSON array of objects with these fields:
- regulation: one of HIPAA, GDPR, EU AI Act, FINRA, CCPA
- company: company or entity fined (string)
- amount_millions: fine amount in millions USD (number, estimate if needed, 0 if unknown)
- reason: brief reason for fine (string, max 10 words)
- year: year of fine (number)

Return ONLY valid JSON array, no markdown, no explanation.
Extract up to 10 most notable fines.

Results:
{text}
"""
    result = llm.invoke([HumanMessage(content=prompt)])
    try:
        clean = result.content.strip().replace("```json", "").replace("```", "")
        return json.loads(clean)
    except Exception:
        return []


def extract_calendar_data(raw_results: list) -> list:
    """Use LLM to extract structured deadline data from raw Tavily results."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    text = "\n\n".join([f"{r['label']}: {r['title']}\n{r['content']}" for r in raw_results])

    prompt = f"""Extract compliance deadline/milestone data from these results.
Return a JSON array of objects with these fields:
- regulation: one of HIPAA, GDPR, EU AI Act, FINRA, CCPA
- deadline: deadline or milestone description (string, max 12 words)
- date: date as string (e.g. "August 2025", "Q1 2026", "January 1, 2026")
- type: one of "Deadline", "Effective Date", "Review", "Enforcement Start"

Return ONLY valid JSON array, no markdown, no explanation.
Extract up to 12 most important upcoming deadlines.

Results:
{text}
"""
    result = llm.invoke([HumanMessage(content=prompt)])
    try:
        clean = result.content.strip().replace("```json", "").replace("```", "")
        return json.loads(clean)
    except Exception:
        return []


def fetch_penalty_data() -> tuple:
    """Fetch and return (raw_results, structured_data) for penalties."""
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    raw = []
    for topic in PENALTY_TOPICS:
        try:
            response = client.search(query=topic["query"], search_depth="basic", max_results=3)
            for r in response.get("results", []):
                raw.append({
                    "label": topic["label"],
                    "title": r.get("title", ""),
                    "content": r.get("content", ""),
                    "url": r.get("url", "#"),
                })
        except Exception:
            pass
    structured = extract_penalty_data(raw)
    return raw, structured


def fetch_calendar_data() -> tuple:
    """Fetch and return (raw_results, structured_data) for calendar."""
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    raw = []
    for topic in CALENDAR_TOPICS:
        try:
            response = client.search(query=topic["query"], search_depth="basic", max_results=3)
            for r in response.get("results", []):
                raw.append({
                    "label": topic["label"],
                    "title": r.get("title", ""),
                    "content": r.get("content", ""),
                    "url": r.get("url", "#"),
                })
        except Exception:
            pass
    structured = extract_calendar_data(raw)
    return raw, structured


# Keep old functions for backward compat
def fetch_penalties(max_results_per_topic=3):
    raw, _ = fetch_penalty_data()
    return raw

def fetch_calendar(max_results_per_topic=3):
    raw, _ = fetch_calendar_data()
    return raw