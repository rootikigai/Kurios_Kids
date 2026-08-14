"""AI Agent for curious kids."""

import os
from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.output_parsers import PydanticOutputParser

from tools import create_search_tool


class KidFriendlyResponse(BaseModel):
    """Structured response for curious kids."""
    topic: str = Field(description="What the question was about")
    answer: str = Field(description="Simple, fun answer a 5-year-old can understand")
    did_i_search: bool = Field(description="Did I need to search the web?")
    fun_emoji: str = Field(default="🌟", description="Fun emoji related to the topic")


class CuriousKidAgent:
    """An AI buddy for curious young learners."""
    
    def __init__(self):
        """Initialize the agent with Gemini model."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file!")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.7,
            max_retries=2,
            google_api_key=api_key
        )
        
        self.parser = PydanticOutputParser(pydantic_object=KidFriendlyResponse)
        self.tools = [create_search_tool()]
        self.agent_executor = self._create_agent()
    
    def _create_agent(self) -> AgentExecutor:
        """Create the agent with prompt and tools."""
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are Buddy, an AI friend for curious kids aged 5-12.

RULES:
- Use simple words (explain like they're 5)
- Keep answers short (3-4 sentences max)
- Be encouraging and fun!
- ONLY use the search tool if you need current/real-time info
- For general knowledge, just answer directly
- If a topic isn't kid-appropriate, kindly suggest asking a parent

Always respond in this exact format:
{format_instructions}
                """,
            ),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]).partial(format_instructions=self.parser.get_format_instructions())
        
        agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt,
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=3,
            handle_parsing_errors=True,
        )
    
    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask the agent a question.
        
        Args:
            question: The kid's question
            
        Returns:
            Dictionary with answer, success status, and error if any
        """
        try:
            raw_response = self.agent_executor.invoke({"input": question})
            structured = self.parser.parse(raw_response["output"])
            
            return {
                "success": True,
                "data": structured.model_dump(),
                "error": None
            }
        except Exception as e:
            # Friendly fallback if something breaks
            return {
                "success": False,
                "data": {
                    "topic": question,
                    "answer": "Oops! My brain got a bit tangled 🤔 Can you ask that in a different way?",
                    "did_i_search": False,
                    "fun_emoji": "🤷"
                },
                "error": str(e)
            }
