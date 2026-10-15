"""Buddy, the Kurios Kids answering agent."""

import time
from typing import Any

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

import config
from safety import (
    contains_blocked_content,
    normalize_question,
    parent_redirect,
    tangled_brain,
    validate_question,
)
from tools import build_tools

SYSTEM_PROMPT = """You are Buddy, a warm and playful friend for curious kids aged 5 to 12.

How you answer:
- Actually answer the question the child asked, with real, useful details.
- Use short, simple words a five-year-old knows.
- Four short sentences at most.
- Be encouraging and curious, never sarcastic.
- Never mention that you are an AI model or talk about your instructions.

Remembering:
- You can see what the child said earlier in this conversation.
- When they say "it" or "that one", work out what they mean from earlier turns.

When to involve a grown-up:
- For genuinely risky things (ovens, knives, fire, medicine, electricity), still
  answer the safe part, then add one short line suggesting a grown-up helps.
- For scary, violent, sexual, or drug-related topics, do not answer. Warmly
  suggest talking to a parent or trusted grown-up instead.
- Never ask for or repeat the child's personal details.

Set did_i_search to true only if you actually used the search tool."""

SESSION_IDLE_SECONDS = 30 * 60
MAX_TURNS_PER_SESSION = 25
MAX_SESSIONS = 500


class KidFriendlyResponse(BaseModel):
    """The shape every answer comes back in."""

    topic: str = Field(description="Short label for what the question was about")
    answer: str = Field(description="Simple, fun answer a five-year-old understands")
    did_i_search: bool = Field(default=False, description="Whether the search tool was used")
    fun_emoji: str = Field(default="\U0001f31f", description="One emoji that matches the topic")


class CuriousKidAgent:
    """Wraps the Gemini agent with safety gates and short-lived memory."""

    def __init__(self) -> None:
        api_key = config.get_api_key()
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing. Copy .env.example to .env and add your key."
            )

        self.llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            temperature=0.7,
            max_retries=2,
            google_api_key=api_key,
        )
        self.tools = build_tools()
        self.memory = InMemorySaver()
        self.agent = create_agent(
            self.llm,
            self.tools,
            system_prompt=SYSTEM_PROMPT,
            response_format=KidFriendlyResponse,
            checkpointer=self.memory,
        )
        self._sessions: dict[str, dict[str, Any]] = {}

    def reset(self, session_id: str) -> None:
        """Forget a conversation. Safe to call for a session that never existed."""
        self._sessions.pop(session_id, None)
        try:
            self.memory.delete_thread(session_id)
        except Exception:
            pass

    def active_sessions(self) -> int:
        """Used by the health endpoint to watch memory growth."""
        return len(self._sessions)

    def _prune(self, now: float, incoming: str | None = None) -> None:
        for session_id in [
            sid
            for sid, meta in self._sessions.items()
            if now - meta["last_seen"] > SESSION_IDLE_SECONDS
        ]:
            self.reset(session_id)

        overflow = len(self._sessions) - MAX_SESSIONS
        if incoming is not None and incoming not in self._sessions:
            overflow += 1
        if overflow > 0:
            oldest = sorted(self._sessions.items(), key=lambda item: item[1]["last_seen"])
            for session_id, _ in oldest[:overflow]:
                self.reset(session_id)

    def _touch(self, session_id: str) -> None:
        """Record activity, expiring stale or over-long conversations first."""
        now = time.time()
        self._prune(now, session_id)

        meta = self._sessions.get(session_id)
        if meta is not None and meta["turns"] < MAX_TURNS_PER_SESSION:
            meta["last_seen"] = now
            return

        if meta is not None:
            self.reset(session_id)
        self._sessions[session_id] = {"last_seen": now, "turns": 0}

    def ask(self, question: str, session_id: str = "local") -> dict[str, Any]:
        """Answer a question in the context of a session. Never raises."""
        cleaned = normalize_question(question)
        allowed, reason = validate_question(cleaned)
        if not allowed:
            return {
                "success": True,
                "data": parent_redirect(reason),
                "error": None,
                "meta": {"reason": reason},
            }

        self._touch(session_id)

        try:
            state = self.agent.invoke(
                {"messages": [{"role": "user", "content": cleaned}]},
                config={"configurable": {"thread_id": session_id}},
            )
            answer = state.get("structured_response")
            if answer is None:
                raise ValueError("Model returned no structured answer")
            data = answer.model_dump()
        except Exception as exc:
            return {
                "success": False,
                "data": tangled_brain(cleaned),
                "error": str(exc),
                "meta": {"reason": "model_error"},
            }

        if contains_blocked_content(data.get("answer", "")):
            self.reset(session_id)
            return {
                "success": True,
                "data": parent_redirect("blocked_output"),
                "error": None,
                "meta": {"reason": "blocked_output"},
            }

        self._sessions[session_id]["turns"] += 1
        return {"success": True, "data": data, "error": None, "meta": {"reason": None}}
