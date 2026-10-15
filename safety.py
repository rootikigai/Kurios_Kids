"""Input and output guards that keep Buddy age-appropriate."""

import re

from config import MAX_QUESTION_CHARS

_BLOCKED = [
    r"\bkill (myself|him|her|them)\b",
    r"\b(suicide|self[\s-]?harm|cutting myself)\b",
    r"\b(sex|sexy|porn|nude|naked|boobs|penis|vagina)\b",
    r"\b(cocaine|heroin|meth|weed|vape|cigarette|beer|vodka)\b",
    r"\b(gun|shoot|bomb|stab)\b",
    r"\b(hate|kill all) (jews|muslims|blacks|whites|gays)\b",
]

_PERSONAL_DATA = [
    r"\bmy (home )?address\b",
    r"\bmy (phone|mobile) number\b",
    r"\bmy password\b",
    r"\bwhere do i live\b",
    r"\bcredit card\b",
]

_INJECTION = [
    r"ignore (all|any|your|the|previous) (instructions|rules|prompt)",
    r"\bsystem prompt\b",
    r"\byou are now\b",
    r"\bpretend you are not\b",
    r"\bdeveloper mode\b",
]

_ALL_INPUT_PATTERNS = _BLOCKED + _PERSONAL_DATA + _INJECTION


def normalize_question(question: str) -> str:
    """Collapse whitespace so length checks and matching are predictable."""
    return " ".join((question or "").split())


def validate_question(question: str) -> tuple[bool, str | None]:
    """Return (is_allowed, reason). Reason is None when allowed."""
    cleaned = normalize_question(question)
    if not cleaned:
        return False, "empty"
    if len(cleaned) > MAX_QUESTION_CHARS:
        return False, "too_long"

    lowered = cleaned.lower()
    for pattern in _ALL_INPUT_PATTERNS:
        if re.search(pattern, lowered):
            return False, "blocked_topic"
    return True, None


def contains_blocked_content(text: str) -> bool:
    """Last line of defence on whatever the model produced."""
    lowered = (text or "").lower()
    return any(re.search(pattern, lowered) for pattern in _BLOCKED)


def parent_redirect(reason: str | None = None) -> dict:
    """Friendly, non-scary deflection that never shames the child."""
    if reason == "empty":
        return {
            "topic": "Ready when you are",
            "answer": "Ask me anything you are curious about! Try 'why do cats purr?'",
            "did_i_search": False,
            "fun_emoji": "\U0001f4ad",
        }
    if reason == "too_long":
        return {
            "topic": "That is a big question",
            "answer": f"Whoa, that was long! Try asking in fewer than {MAX_QUESTION_CHARS} letters.",
            "did_i_search": False,
            "fun_emoji": "\U0001f4cf",
        }
    return {
        "topic": "Ask a grown-up",
        "answer": "That one is best for a parent or a trusted grown-up. Let's pick something else fun to explore!",
        "did_i_search": False,
        "fun_emoji": "\U0001f6e1",
    }


def tangled_brain(question: str) -> dict:
    """Shown when the model or network fails. Never expose the error to a child."""
    return {
        "topic": normalize_question(question)[:60] or "Hmm",
        "answer": "Oops, my brain got a bit tangled. Can you ask me that a different way?",
        "did_i_search": False,
        "fun_emoji": "\U0001f937",
    }
