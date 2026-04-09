def format_response(raw_response: str) -> str:
    """Clean and format the raw LLM response for display."""
    if not raw_response:
        return "No response received."
    return raw_response.strip()