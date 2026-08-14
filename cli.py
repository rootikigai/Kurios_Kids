"""Simple test script for the agent."""

from dotenv import load_dotenv
from agent import CuriousKidAgent

# Load environment variables
load_dotenv()

def main():
    print("🌟 Starting Curious Kid AI...\n")
    
    # Create agent
    try:
        agent = CuriousKidAgent()
        print("✅ Agent initialized successfully!\n")
    except Exception as e:
        print(f"❌ Error initializing agent: {e}")
        return
    
    # Interactive loop
    print("Type 'quit' to exit\n")
    
    while True:
        question = input("🤔 What are you curious about today?: ")
        
        if question.lower() in ['quit', 'exit', 'bye']:
            print("\n👋 Bye! Keep being curious!")
            break
        
        if not question.strip():
            continue
        
        print("\n💭 Thinking...\n")
        result = agent.ask(question)
        
        if result["success"]:
            data = result["data"]
            print(f"{data['fun_emoji']} {data['topic']}")
            print(f"\n{data['answer']}\n")
            if data["did_i_search"]:
                print("🔍 (I searched the web to find this!)\n")
        else:
            print(f"❌ {result['data']['answer']}\n")
            print(f"Error details: {result['error']}\n")

if __name__ == "__main__":
    main()
