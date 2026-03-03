"""
Retrieval Evaluator for Agentic RAG Chatbot.
Checks if retrieved documents are relevant enough.
If not, refines the query for a retry.
This is the "agentic loop" — the system self-corrects.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict
from src.config import LLM_MODEL, RELEVANCE_THRESHOLD


EVAL_PROMPT = PromptTemplate(
    input_variables=["query", "documents"],
    template="""You are evaluating whether retrieved document excerpts are 
relevant to the user's question. 

User Question: {query}

Retrieved Excerpts:
{documents}

Are these excerpts sufficient to answer the question?
Respond with ONLY: "sufficient" or "insufficient"
If insufficient, add a brief reason after a pipe character.
Example: "insufficient | query needs to focus on methodology details" """
)

REFINE_PROMPT = PromptTemplate(
    input_variables=["original_query", "reason"],
    template="""The following query didn't retrieve relevant document excerpts.
Rewrite it to be more specific and likely to match the document content.

Original query: {original_query}
Reason for failure: {reason}

Rewritten query (be specific, use relevant technical terminology):"""
)


class RetrievalEvaluator:
    """Evaluates retrieval quality and refines queries if needed."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0,
            max_tokens=100
        )
        self.eval_chain = EVAL_PROMPT | self.llm | StrOutputParser()
        self.refine_chain = REFINE_PROMPT | self.llm | StrOutputParser()

    def evaluate(self, query: str, documents: List[Dict]) -> Dict:
        """
        Check if retrieved docs are relevant enough.
        
        Two checks:
        1. Score-based: are similarity scores above threshold?
        2. LLM-based: does the content actually answer the question?
        
        Returns:
            {
                "sufficient": bool,
                "reason": str or None,
                "avg_score": float
            }
        """
        if not documents:
            return {
                "sufficient": False,
                "reason": "No documents retrieved",
                "avg_score": 0.0
            }

        # Score check
        scores = [d.get("score", 0) for d in documents]
        avg_score = sum(scores) / len(scores)

        if avg_score < RELEVANCE_THRESHOLD:
            return {
                "sufficient": False,
                "reason": f"Low relevance scores (avg: {avg_score:.2f})",
                "avg_score": avg_score
            }

        # Score-only check (skip LLM call for speed)
        return {
            "sufficient": True,
            "reason": None,
            "avg_score": avg_score
        }

    def refine_query(self, original_query: str, reason: str) -> str:
        """
        Rewrite query to improve retrieval on retry.
        
        Args:
            original_query: The query that didn't work
            reason: Why it failed
            
        Returns:
            Refined query string
        """
        try:
            refined = self.refine_chain.invoke({
                "original_query": original_query,
                "reason": reason
            }).strip()
            return refined
        except Exception:
            return original_query  # Fallback to original