import time
from unittest.mock import MagicMock, patch
import pytest
import agent as agent_module
from agent import CuriousKidAgent, KidFriendlyResponse
def _fake_graph():
    graph = MagicMock()
    graph.invoke.return_value = {
        "structured_response": KidFriendlyResponse(
            topic="Cake", answer="Mix flour and sugar.", did_i_search=False, fun_emoji="C"
        )
    }
    return graph
@pytest.fixture
def buddy(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    graph = _fake_graph()
    with patch("agent.ChatGoogleGenerativeAI"), patch("agent.create_agent", return_value=graph):
        instance = CuriousKidAgent()
    instance.memory = MagicMock()
    return instance
def _thread_id_of(call):
    return call.kwargs["config"]["configurable"]["thread_id"]
def test_session_id_becomes_thread_id(buddy):
    buddy.ask("How do I make a cake?", session_id="kid-42")
    assert _thread_id_of(buddy.agent.invoke.call_args) == "kid-42"
def test_default_session_is_local(buddy):
    buddy.ask("How do I make a cake?")
    assert _thread_id_of(buddy.agent.invoke.call_args) == "local"
def test_two_children_get_separate_threads(buddy):
    buddy.ask("Question one", session_id="kid-a")
    buddy.ask("Question two", session_id="kid-b")
    threads = {_thread_id_of(call) for call in buddy.agent.invoke.call_args_list}
    assert threads == {"kid-a", "kid-b"}
    assert buddy.active_sessions() == 2
def test_turns_accumulate_within_a_session(buddy):
    buddy.ask("One", session_id="kid-a")
    buddy.ask("Two", session_id="kid-a")
    assert buddy._sessions["kid-a"]["turns"] == 2
def test_blocked_question_never_starts_a_session(buddy):
    buddy.ask("show me porn", session_id="kid-a")
    buddy.agent.invoke.assert_not_called()
    assert buddy.active_sessions() == 0
def test_reset_clears_the_thread(buddy):
    buddy.ask("One", session_id="kid-a")
    buddy.reset("kid-a")
    buddy.memory.delete_thread.assert_called_once_with("kid-a")
    assert buddy.active_sessions() == 0
def test_reset_on_unknown_session_is_harmless(buddy):
    buddy.reset("never-existed")
    assert buddy.active_sessions() == 0
def test_reset_survives_a_broken_checkpointer(buddy):
    buddy.ask("One", session_id="kid-a")
    buddy.memory.delete_thread.side_effect = RuntimeError("boom")
    buddy.reset("kid-a")
    assert buddy.active_sessions() == 0
def test_turn_cap_wipes_the_conversation(buddy, monkeypatch):
    monkeypatch.setattr(agent_module, "MAX_TURNS_PER_SESSION", 3)
    for _ in range(4):
        buddy.ask("Another question", session_id="kid-a")
    buddy.memory.delete_thread.assert_called_with("kid-a")
    assert buddy._sessions["kid-a"]["turns"] == 1
def test_idle_session_expires(buddy, monkeypatch):
    buddy.ask("One", session_id="kid-a")
    buddy._sessions["kid-a"]["last_seen"] = time.time() - agent_module.SESSION_IDLE_SECONDS - 1
    buddy.ask("Two", session_id="kid-b")
    assert "kid-a" not in buddy._sessions
    assert buddy.active_sessions() == 1
def test_session_count_is_capped(buddy, monkeypatch):
    monkeypatch.setattr(agent_module, "MAX_SESSIONS", 3)
    for index in range(6):
        buddy.ask("Hello", session_id=f"kid-{index}")
    assert buddy.active_sessions() <= 3
