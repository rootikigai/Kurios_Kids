"""Interactive CLI for trying Buddy locally."""

from agent import CuriousKidAgent

SESSION_ID = "local"
RESET_WORDS = {"reset", "start over", "forget", "new chat"}
QUIT_WORDS = {"quit", "exit", "bye"}


def main() -> None:
    print("\n\U0001f31f Kurios Kids - Buddy is waking up...\n")

    try:
        buddy = CuriousKidAgent()
    except ValueError as exc:
        print(f"\u274c {exc}\n")
        return

    print("\u2705 Buddy is ready.")
    print("   Type 'reset' to start a fresh chat, or 'quit' to leave.\n")

    while True:
        try:
            question = input("\U0001f914 What are you curious about? ")
        except (EOFError, KeyboardInterrupt):
            print("\n\n\U0001f44b Bye! Keep being curious!\n")
            return

        command = question.strip().lower()

        if command in QUIT_WORDS:
            print("\n\U0001f44b Bye! Keep being curious!\n")
            return

        if command in RESET_WORDS:
            buddy.reset(SESSION_ID)
            print("\n\U0001f9f9 Okay, fresh start! I forgot what we talked about.\n")
            continue

        print("\n\U0001f4ad Thinking...\n")
        result = buddy.ask(question, session_id=SESSION_ID)
        data = result["data"]

        print(f"{data['fun_emoji']}  {data['topic']}")
        print(f"\n{data['answer']}\n")

        if data["did_i_search"]:
            print("\U0001f50d (I looked this up on the web.)\n")
        if not result["success"]:
            print(f"[debug] {result['error']}\n")


if __name__ == "__main__":
    main()
