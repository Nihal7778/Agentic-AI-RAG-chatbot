"""
Memory Writer for Agentic RAG Chatbot.
Decides what to save and writes to USER_MEMORY.md / COMPANY_MEMORY.md.
Only writes high-signal, selective facts — no transcript dumping.
"""

import json
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict
from src.config import (
    LLM_MODEL, MEMORY_CONFIDENCE_THRESHOLD,
    USER_MEMORY_PATH, COMPANY_MEMORY_PATH
)


MEMORY_DECISION_PROMPT = PromptTemplate(
    input_variables=["query", "response", "existing_user_memory", "documents"],
    template="""You are a memory manager for a document Q&A chatbot.
Decide if anything from this interaction is worth remembering.

User's Question: {query}
Bot's Response (summary): {response}
Existing User Memory: {existing_user_memory}
Document Sections Referenced: {documents}

Decide what to remember. Respond in this JSON format ONLY (no markdown):
{{
    "user_memory": {{
        "should_write": true/false,
        "summary": "fact about the user" or null,
        "confidence": 0.0 to 1.0
    }},
    "company_memory": {{
        "should_write": true/false,
        "summary": "reusable learning from the document" or null,
        "confidence": 0.0 to 1.0
    }}
}}

Rules:
- User memory: interests, expertise level, research focus, preferences
  e.g. "User is interested in deep learning for stock prediction"
  e.g. "User prefers simplified explanations of math concepts"
- Company memory: reusable knowledge from documents analyzed
  e.g. "CNN models are faster than LSTM but have comparable accuracy for stock prediction"
  e.g. "Ensemble methods improve medical image classification accuracy"
- Only write if confidence > 0.7
- Don't duplicate what's already in existing memory"""
)


class MemoryWriter:
    """Decides what to save and writes to memory files."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0,
            max_tokens=300
        )
        self.chain = MEMORY_DECISION_PROMPT | self.llm | StrOutputParser()

    def decide_and_write(
        self,
        query: str,
        response: str,
        documents: List[Dict],
        user_context: str = ""
    ) -> Dict:
        """
        Decide if anything should be saved, then write if so.

        Returns:
            {"wrote_user": bool, "wrote_company": bool, "details": str}
        """
        result = {"wrote_user": False, "wrote_company": False, "details": ""}

        try:
            doc_summary = "; ".join(
                f"{d['metadata'].get('section_type', '?')}: {d['metadata'].get('section_title', '?')}"
                for d in documents[:5]
            )

            decision_raw = self.chain.invoke({
                "query": query,
                "response": response[:500],
                "existing_user_memory": user_context[:300],
                "documents": doc_summary
            })

            decision = self._parse_decision(decision_raw)

            # Write user memory
            user_mem = decision.get("user_memory", {})
            if (user_mem.get("should_write") and
                user_mem.get("confidence", 0) >= MEMORY_CONFIDENCE_THRESHOLD and
                user_mem.get("summary")):
                self._append_memory(USER_MEMORY_PATH, user_mem["summary"])
                result["wrote_user"] = True

            # Write company memory
            company_mem = decision.get("company_memory", {})
            if (company_mem.get("should_write") and
                company_mem.get("confidence", 0) >= MEMORY_CONFIDENCE_THRESHOLD and
                company_mem.get("summary")):
                self._append_memory(COMPANY_MEMORY_PATH, company_mem["summary"])
                result["wrote_company"] = True

        except Exception as e:
            result["details"] = f"Memory write failed: {e}"

        return result

    def _append_memory(self, path, summary: str):
        """Append a memory entry with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry = f"\n- [{timestamp}] {summary}"

        try:
            if not path.exists():
                header = "# User Memory\n" if "USER" in str(path) else "# Company Memory\n"
                path.write_text(header, encoding="utf-8")

            with open(path, "a", encoding="utf-8") as f:
                f.write(entry + "\n")
        except Exception as e:
            print(f"⚠️ Failed to write memory to {path}: {e}")

    def _parse_decision(self, raw: str) -> Dict:
        """Parse LLM JSON response with fallback."""
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {}