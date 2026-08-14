from langchain_core.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun

def create_search_tool() -> Tool:
    """Create a kid-friendly web search tool."""
    search = DuckDuckGoSearchRun()

    return Tool(
        name="searchWeb",
        func=lambda query: search.run(query + " simple explanation for kids"),
        description="Search the web for kid-friendly information. Only use when the question needs current/fresh data.",
    )
