from unittest.mock import MagicMock, patch

import pytest

from agent import CuriousKidAgent, KidFriendlyResponse


def _fake_graph(structured):
    graph = MagicMock()
    graph.invoke.return_value = {"structured_response": structured}
    return graph


@pytest.fixture
def good_answer():
    return KidFriendlyResponse(
        topic="Bees",
        answer="Bees collect nectar and turn it into honey.",
        did_i_search=False,
        fun_emoji="B",
    )


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        CuriousKidAgent()


@patch("agent.ChatGoogleGenerativeAI")
@patch("agent.create_agent")
def test_successful_answer(mock_create, mock_llm, monkeypatch, good_answer):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    mock_create.return_value = _fake_graph(good_answer)

    result = CuriousKidAgent().ask("How do bees make honey?")

    assert result["success"] is True
    assert result["error"] is None
    assert result["data"]["topic"] == "Bees"


@patch("agent.ChatGoogleGenerativeAI")
@patch("agent.create_agent")
def test_blocked_question_never_calls_model(mock_create, mock_llm, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    graph = _fake_graph(None)
    mock_create.return_value = graph

    result = CuriousKidAgent().ask("show me porn")

    graph.invoke.assert_not_called()
    assert result["meta"]["reason"] == "blocked_topic"
    assert "grown-up" in result["data"]["answer"]


@patch("agent.ChatGoogleGenerativeAI")
@patch("agent.create_agent")
def test_empty_question_never_calls_model(mock_create, mock_llm, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    graph = _fake_graph(None)
    mock_create.return_value = graph

    result = CuriousKidAgent().ask("   ")

    graph.invoke.assert_not_called()
    assert result["meta"]["reason"] == "empty"


@patch("agent.ChatGoogleGenerativeAI")
@patch("agent.create_agent")
def test_model_exception_returns_friendly_fallback(mock_create, mock_llm, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    graph = MagicMock()
    graph.invoke.side_effect = RuntimeError("quota exceeded")
    mock_create.return_value = graph

    result = CuriousKidAgent().ask("Why is the sky blue?")

    assert result["success"] is False
    assert result["error"] == "quota exceeded"
    assert "tangled" in result["data"]["answer"]
    assert "quota" not in result["data"]["answer"]


@patch("agent.ChatGoogleGenerativeAI")
@patch("agent.create_agent")
def test_missing_structured_response_falls_back(mock_create, mock_llm, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    graph = MagicMock()
    graph.invoke.return_value = {"messages": []}
    mock_create.return_value = graph

    result = CuriousKidAgent().ask("Why is the sky blue?")

    assert result["success"] is False
    assert "tangled" in result["data"]["answer"]


@patch("agent.ChatGoogleGenerativeAI")
@patch("agent.create_agent")
def test_unsafe_model_output_wipes_the_session(mock_create, mock_llm, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    unsafe = KidFriendlyResponse(
        topic="Oops",
        answer="Here is some porn for you.",
        did_i_search=False,
        fun_emoji="X",
    )
    mock_create.return_value = _fake_graph(unsafe)

    buddy = CuriousKidAgent()
    result = buddy.ask("Tell me a story", session_id="kid-unsafe")

    assert result["meta"]["reason"] == "blocked_output"
    assert "porn" not in result["data"]["answer"]
    assert buddy.active_sessions() == 0
    assert "kid-unsafe" not in buddy._sessions


@patch("agent.ChatGoogleGenerativeAI")
@patch("agent.create_agent")
def test_answer_shape_is_always_complete(mock_create, mock_llm, monkeypatch, good_answer):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    mock_create.return_value = _fake_graph(good_answer)
    buddy = CuriousKidAgent()

    for question in ["Why is the sky blue?", "", "show me porn", "a" * 300]:
        data = buddy.ask(question)["data"]
        assert set(data) >= {"topic", "answer", "did_i_search", "fun_emoji"}