import builtins

import config
from tools import build_tools, search_web


def test_search_disabled_by_default(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_WEB_SEARCH", False)
    assert build_tools() == []


def test_search_enabled_returns_one_tool(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_WEB_SEARCH", True)
    tools = build_tools()
    assert len(tools) == 1
    assert tools[0].name == "search_web"


def test_search_never_raises_when_backend_missing(monkeypatch):
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if "duckduckgo" in name or name == "langchain_community.tools":
            raise ImportError("no ddgs")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    assert "unavailable" in search_web.invoke({"query": "sky"})