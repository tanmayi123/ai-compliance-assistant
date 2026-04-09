from datetime import datetime
from tavily import TavilyClient
import os


# Topics to search for updates
UPDATE_TOPICS = [
    {"label": "🏥 HIPAA", "query": "HIPAA compliance updates regulations 2025"},
    {"label": "🇪🇺 GDPR", "query": "GDPR enforcement updates fines 2025"},
    {"label": "🤖 EU AI Act", "query": "EU AI Act implementation updates 2025"},
    {"label": "💰 FINRA", "query": "FINRA financial compliance rule updates 2025"},
    {"label": "🔒 CCPA", "query": "CCPA California privacy law updates 2025"},
]


def fetch_law_updates(max_results_per_topic: int = 3) -> list[dict]:
    """
    Search Tavily for recent updates on each compliance topic.
    Returns a list of result dicts with label, title, content, url, and timestamp.
    """
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    all_results = []

    for topic in UPDATE_TOPICS:
        try:
            response = client.search(
                query=topic["query"],
                search_depth="basic",
                max_results=max_results_per_topic
            )
            for result in response.get("results", []):
                all_results.append({
                    "label": topic["label"],
                    "title": result.get("title", "No title"),
                    "content": result.get("content", "No summary available."),
                    "url": result.get("url", "#"),
                    "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M")
                })
        except Exception as e:
            all_results.append({
                "label": topic["label"],
                "title": "Error fetching updates",
                "content": str(e),
                "url": "#",
                "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M")
            })

    return all_results