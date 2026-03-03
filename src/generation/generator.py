"""
Response Generator for Agentic RAG Chatbot.
Builds grounded answers with citations from scored documents.
Supports conversation history for follow-up questions.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict
from src.config import LLM_MODEL, LLM_TEMPERATURE, COMPLEXITY_LEVELS


GENERATE_PROMPT = PromptTemplate(
    input_variables=["question", "context", "conversation_history"],
    template=(
        "You are a retrieval-grounded assistant for question answering.\n"
        "Answer ONLY using the provided context. Do NOT use outside knowledge.\n"
        "As a knowledgeable and helpful research assistant, your task is to provide informative answers based on the given context. "
        "You are a research document analyst. Given a question about a document, write a short passage (2-3 sentences) that would appear in a research paper addressing this topic. Use formal academic language."
        "Use your extensive knowledge base to offer clear, concise, and accurate responses to the user's inquiries.\n"
        "If any question is asked outside of the provided documents then display "
        "\"I don't have enough information in the provided documents to answer that accurately\".\n\n"
        "Style rules:\n"
        "- Use 1–3 short sentences.\n"
        "- Prefer wording copied or closely paraphrased from the context (be extractive).\n"
        "- Include the key terms from the question in your answer.\n"
        "- Do not add extra background beyond what the context supports.\n"
        "- Cite sources using [Section X, Page Y] format.\n\n"
        "Previous conversation:\n{conversation_history}\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    )
)


class ResponseGenerator:
    """Generates grounded answers with citations."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=1500
        )
        self.chain = GENERATE_PROMPT | self.llm | StrOutputParser()

    def generate(
        self,
        query: str,
        documents: List[Dict],
        user_context: str = "",
        company_context: str = "",
        conversation_history: str = "",
        strategy: str = "complex"
    ) -> Dict:
        """
        Generate a grounded response with citations.

        Returns:
            {
                "answer": str,
                "citations": list,
                "complexity_summary": {"high": int, "medium": int, "low": int}
            }
        """
        if not documents:
            return {
                "answer": "I don't have any documents to reference. Please upload a document first.",
                "citations": [],
                "complexity_summary": {"high": 0, "medium": 0, "low": 0}
            }

        # Filter out low-relevance documents
        threshold = 0 if strategy == "simple" else 0.2
        relevant_docs = [d for d in documents if d.get("score", 0) >= threshold]

        if not relevant_docs:
            return {
                "answer": "I don't have enough information in the provided documents to answer that accurately.",
                "citations": [],
                "complexity_summary": {"high": 0, "medium": 0, "low": 0}
            }

        context = self._build_context(relevant_docs)
        citations = self._extract_citations(relevant_docs)

        try:
            answer = self.chain.invoke({
                "question": query,
                "context": context,
                "conversation_history": conversation_history or "No previous conversation."
            })
        except Exception as e:
            answer = f"Error generating response: {e}"

        complexity_summary = {
            "high": sum(1 for d in relevant_docs if d.get("complexity") == "high"),
            "medium": sum(1 for d in relevant_docs if d.get("complexity") == "medium"),
            "low": sum(1 for d in relevant_docs if d.get("complexity") == "low"),
        }

        return {
            "answer": answer,
            "citations": citations,
            "complexity_summary": complexity_summary
        }

    def _build_context(self, documents: List[Dict]) -> str:
        """Format documents into context string with citations."""
        parts = []
        for i, doc in enumerate(documents, 1):
            cmplx = COMPLEXITY_LEVELS.get(doc.get("complexity", "low"), "⚪")
            sec = doc["metadata"].get("section_number", "?")
            page = doc["metadata"].get("page_number", "?")
            stype = doc["metadata"].get("section_type", "general")
            title = doc["metadata"].get("section_title", "")

            header = f"[{i}] Section {sec}, Page {page} ({stype})"
            if title:
                header += f" — \"{title[:60]}\""

            parts.append(f"{header}\n{doc['text'][:600]}")

        return "\n\n---\n\n".join(parts)

    def _build_complexity_summary(self, documents: List[Dict]) -> str:
        """Build a quick complexity summary string."""
        high = [d for d in documents if d.get("complexity") == "high"]
        med = [d for d in documents if d.get("complexity") == "medium"]
        low = [d for d in documents if d.get("complexity") == "low"]

        lines = []
        if high:
            lines.append(f"🔴 HIGH COMPLEXITY: {len(high)} section(s)")
        if med:
            lines.append(f"🟡 MEDIUM COMPLEXITY: {len(med)} section(s)")
        if low:
            lines.append(f"🟢 LOW COMPLEXITY: {len(low)} section(s)")

        return "\n".join(lines) if lines else "No complexity data available."

    def _extract_citations(self, documents: List[Dict]) -> List[Dict]:
        """Extract structured citation data from documents."""
        citations = []
        for doc in documents:
            citations.append({
                "section": doc["metadata"].get("section_number", "?"),
                "page": doc["metadata"].get("page_number", "?"),
                "section_type": doc["metadata"].get("section_type", "general"),
                "section_title": doc["metadata"].get("section_title", ""),
                "complexity": doc.get("complexity", "low"),
                "chunk_id": doc["metadata"].get("chunk_id", "")
            })
        return citations