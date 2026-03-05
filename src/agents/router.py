"""
Query Router for Agentic RAG Chatbot.
Classifies incoming queries to decide retrieval strategy.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import LLM_MODEL, MULTIMODAL_ENABLED
from src.tools.weather import WeatherTool


ROUTER_PROMPT = PromptTemplate(
    input_variables=["query"],
    template="""You are a query classifier for a document Q&A chatbot.
Classify the following user query into exactly ONE category:

- "conversational" → Greetings, small talk, or questions unrelated to any document.
  Examples: "hey", "hello", "how are you?", "thanks", "what can you do?"

- "simple" → User asks about a specific section, chapter, page, or wants a direct lookup.
  Examples: "What does Chapter 3 say?", "Show me the results section", "What's on page 45?"

- "complex" → User asks analytical, comparative, or conceptual questions about a document.
  Examples: "Compare CNN and LSTM performance", "What methodology did they use?"

- "image_query" → User asks about figures, charts, diagrams, graphs, or visual content.
  Examples: "Show me the architecture diagram", "What does Figure 3 show?", "Describe the performance chart"

Query: {query}

Respond with ONLY one word: conversational, simple, complex, or image_query"""
)

# Fast keyword check before LLM call
CONVERSATIONAL_KEYWORDS = [
    "hey", "hello", "hi", "howdy", "sup", "yo",
    "how are you", "how's it going", "what's up", "how do you do",
    "good morning", "good evening", "good afternoon",
    "thanks", "thank you", "bye", "goodbye", "see you",
    "what can you do", "who are you", "help me",
]

SIMPLE_KEYWORDS = [
    "section", "chapter", "page",
    "appendix", "reference", "read", "show me",
    "what does", "find the", "look up", "quote",
    "paragraph", "line"
]

IMAGE_KEYWORDS = [
    "image", "figure", "diagram", "chart", "graph", "plot",
    "picture", "illustration", "visual", "photo",
    "show me the figure", "what does the diagram",
    "describe the chart", "architecture diagram",
    "show me the graph", "what does figure",
]

TECHNICAL_KEYWORDS = [
    "compare", "explain", "how does", "why", "analyze",
    "summarize", "methodology", "algorithm", "model",
    "performance", "accuracy", "evaluate", "difference",
    "advantage", "disadvantage", "limitation", "approach",
    "propose", "conclude", "finding", "result",
    "neural network", "deep learning", "machine learning",
    "training", "prediction", "classification"
]


class QueryRouter:
    """Routes queries to the appropriate retrieval strategy."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0,
            max_tokens=10
        )
        self.chain = ROUTER_PROMPT | self.llm | StrOutputParser()

    def classify(self, query: str) -> dict:
        """
        Classify a query as conversational, simple, complex, image_query, or tool_call.

        Returns:
            {
                "strategy": "conversational" | "simple" | "complex" | "image_query" | "tool_call",
                "is_technical_query": bool,
                "method": "keyword" | "llm"
            }
        """
        q_lower = query.lower().strip()

        # Weather/tool check first
        weather_location = WeatherTool.detect_weather_query(query)
        if weather_location:
            return {
                "strategy": "tool_call",
                "is_technical_query": False,
                "method": "keyword",
                "tool": "weather",
                "location": weather_location
            }

        # Conversational check
        if any(kw in q_lower for kw in CONVERSATIONAL_KEYWORDS):
            has_doc_words = any(
                kw in q_lower for kw in SIMPLE_KEYWORDS + TECHNICAL_KEYWORDS + IMAGE_KEYWORDS
            )
            if not has_doc_words:
                return {
                    "strategy": "conversational",
                    "is_technical_query": False,
                    "method": "keyword"
                }

        # Image query check (before simple, since "figure" and "table" overlap)
        if MULTIMODAL_ENABLED and any(kw in q_lower for kw in IMAGE_KEYWORDS):
            return {
                "strategy": "image_query",
                "is_technical_query": False,
                "method": "keyword"
            }

        if any(kw in q_lower for kw in SIMPLE_KEYWORDS):
            return {
                "strategy": "simple",
                "is_technical_query": False,
                "method": "keyword"
            }

        if any(kw in q_lower for kw in TECHNICAL_KEYWORDS):
            return {
                "strategy": "complex",
                "is_technical_query": True,
                "method": "keyword"
            }

        # LLM classification
        try:
            result = self.chain.invoke({"query": query}).strip().lower()
            if "image" in result and MULTIMODAL_ENABLED:
                strategy = "image_query"
            elif "conversational" in result:
                strategy = "conversational"
            elif "simple" in result:
                strategy = "simple"
            else:
                strategy = "complex"
            return {
                "strategy": strategy,
                "is_technical_query": strategy == "complex",
                "method": "llm"
            }
        except Exception as e:
            print(f"⚠️ Router LLM failed: {e}, defaulting to complex")
            return {
                "strategy": "complex",
                "is_technical_query": False,
                "method": "fallback"
            }