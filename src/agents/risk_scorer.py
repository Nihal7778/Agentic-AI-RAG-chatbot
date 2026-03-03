"""
Complexity Scorer for Agentic RAG Chatbot.
Analyzes retrieved document sections and rates their complexity.
Helps users understand the technical depth of the content.
"""

import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict
from src.config import (
    LLM_MODEL, LLM_TEMPERATURE,
    HIGH_COMPLEXITY_SECTIONS, MEDIUM_COMPLEXITY_SECTIONS
)


COMPLEXITY_PROMPT = PromptTemplate(
    input_variables=["section_text", "section_type", "user_query"],
    template="""You are a document analyst. Assess the complexity
and relevance of this document excerpt for answering the user's question.

Section Type: {section_type}
User Question: {user_query}

Excerpt:
{section_text}

Respond in this exact JSON format only (no markdown, no backticks):
{{
    "complexity": "high" or "medium" or "low",
    "relevance_explanation": "How this excerpt helps answer the question (2-3 sentences)",
    "key_concepts": ["concept 1", "concept 2"],
    "suggested_followup": "A follow-up question the user might want to ask" or null
}}"""
)


class ComplexityScorer:
    """Scores document sections for complexity and relevance."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=400
        )
        self.chain = COMPLEXITY_PROMPT | self.llm | StrOutputParser()

    def score_sections(
        self,
        documents: List[Dict],
        user_query: str = ""
    ) -> List[Dict]:
        """
        Score each retrieved section for complexity and relevance.

        Uses a two-pass approach:
        1. Quick pre-score based on section type metadata
        2. LLM deep analysis for high/medium complexity sections

        Args:
            documents: Retrieved docs from retrieval pipeline
            user_query: The user's original question

        Returns:
            List of scored documents with analysis
        """
        scored = []

        for doc in documents:
            section_type = doc["metadata"].get("section_type", "general")
            pre_score = self._pre_score(section_type)

            if pre_score == "low":
                scored.append({
                    **doc,
                    "complexity": "low",
                    "relevance_explanation": "This section contains general or introductory content.",
                    "key_concepts": [],
                    "suggested_followup": None
                })
                continue

            # Skip LLM call for speed — use pre-score only
            scored.append({
                **doc,
                "complexity": pre_score,
                "relevance_explanation": f"This {section_type} section is relevant to your query.",
                "key_concepts": [],
                "suggested_followup": None
            })

        return scored

    def _pre_score(self, section_type: str) -> str:
        """Quick complexity pre-classification based on section type."""
        if section_type in HIGH_COMPLEXITY_SECTIONS:
            return "high"
        if section_type in MEDIUM_COMPLEXITY_SECTIONS:
            return "medium"
        return "low"

    def _parse_response(self, response: str) -> Dict:
        """Parse LLM JSON response with fallback."""
        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {
                "complexity": "medium",
                "relevance_explanation": response[:300],
                "key_concepts": [],
                "suggested_followup": None
            }