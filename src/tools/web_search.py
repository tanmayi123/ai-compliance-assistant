from langchain_community.tools.tavily_search import TavilySearchResults


def get_web_search_tool(max_results: int = 3):
    """
    Return a LangChain-compatible Tavily web search tool.
    Used by the agent when RAG confidence is low or query is about recent updates.
    """
    tool = TavilySearchResults(
        max_results=max_results,
        description=(
            "Use this tool to search the web for recent compliance law updates, "
            "regulations not covered in the document store, or current legal news. "
            "Input should be a clear search query."
        )
    )
    return tool