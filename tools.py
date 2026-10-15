"""Optional web search. Disabled by default for the MVP."""

from langchain_core.tools import BaseTool, tool

import config

_MAX_RESULT_CHARS = 1200


@tool
def search_web(query: str) -> str:
    """Look up current facts such as today's weather or recent events.

    Only use this for information that changes over time. Answer everything
    else from your own knowledge.
    """
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
    except ImportError:
        return "Search is unavailable right now."

    try:
        results = DuckDuckGoSearchRun().run(f"{query} explained simply for children")
    except Exception:
        return "Search is unavailable right now."

    return (results or "No results found.")[:_MAX_RESULT_CHARS]


def build_tools() -> list[BaseTool]:
    """Return the tool list for the agent, honouring the feature flag."""
    return [search_web] if config.ENABLE_WEB_SEARCH else []