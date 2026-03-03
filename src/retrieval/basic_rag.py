"""
Basic RAG retrieval for simple/direct queries.
Uses metadata filtering for section/chapter/type lookups,
falls back to semantic search otherwise.
"""

import re
import time
from typing import List, Dict, Optional
from src.retrieval.embedder import ContractEmbedder
from src.config import TOP_K


SECTION_TYPE_KEYWORDS = {
    "introduction": ["introduction", "intro", "overview"],
    "methodology": ["methodology", "methods", "approach", "proposed"],
    "results": ["results", "findings", "performance", "experimental"],
    "discussion": ["discussion", "implications"],
    "conclusion": ["conclusion", "summary", "concluding"],
    "references": ["references", "bibliography", "citations"],
    "related_work": ["related work", "literature", "previous work"],
    "background": ["background", "preliminaries"],
    "abstract": ["abstract"],
}


class BasicRAGRetriever:
    """Simple retrieval with smart section/type detection."""

    def __init__(self, embedder: ContractEmbedder):
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        k: int = TOP_K,
        filter_dict: Optional[Dict] = None
    ) -> Dict:
        start = time.time()

        # Try 1: section/chapter number lookup
        section_num = self._extract_section_number(query)
        if section_num:
            docs = self._search_by_section(section_num, k, filter_dict)
            if docs:
                return {
                    "documents": docs,
                    "retrieval_time": time.time() - start,
                    "strategy": "basic_section_number"
                }

        # Try 2: section type lookup (introduction, results, etc.)
        section_type = self._extract_section_type(query)
        if section_type:
            docs = self._search_by_type(section_type, k, filter_dict)
            if docs:
                return {
                    "documents": docs,
                    "retrieval_time": time.time() - start,
                    "strategy": "basic_section_type"
                }

        # Try 3: keyword in section title
        title_docs = self._search_by_title(query, k, filter_dict)
        if title_docs:
            return {
                "documents": title_docs,
                "retrieval_time": time.time() - start,
                "strategy": "basic_title_match"
            }

        # Fallback: semantic search
        docs = self.embedder.search(query=query, k=k, filter_dict=filter_dict)

        return {
            "documents": docs,
            "retrieval_time": time.time() - start,
            "strategy": "basic_semantic"
        }

    def _extract_section_number(self, query: str) -> Optional[str]:
        """Extract section/chapter number from query."""
        q = query.lower()
        patterns = [
            r'chapter\s*(\d+)',
            r'section\s*(\d+(?:\.\d+)*)',
            r'page\s*(\d+)',
        ]
        for p in patterns:
            m = re.search(p, q)
            if m:
                return m.group(1)
        return None

    def _extract_section_type(self, query: str) -> Optional[str]:
        """Extract section type from query (introduction, results, etc.)."""
        q = query.lower()
        for sec_type, keywords in SECTION_TYPE_KEYWORDS.items():
            if any(kw in q for kw in keywords):
                return sec_type
        return None

    def _search_by_section(self, section_num: str, k: int, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """Search by section number metadata."""
        try:
            # Direct match
            where = {"section_number": section_num}
            if filter_dict:
                where = {"$and": [where, filter_dict]}

            results = self.embedder.collection.get(
                where=where, limit=k, include=["documents", "metadatas"]
            )

            if results["ids"]:
                return self._format_results(results, 0.9)

            # Partial match — section starts with the number
            return self._scan_metadata("section_number", section_num, k)

        except Exception as e:
            print(f"⚠️ Section search failed: {e}")
            return []

    def _search_by_type(self, section_type: str, k: int, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """Search by section type metadata."""
        try:
            where = {"section_type": section_type}
            if filter_dict:
                where = {"$and": [where, filter_dict]}

            results = self.embedder.collection.get(
                where=where, limit=k, include=["documents", "metadatas"]
            )

            if results["ids"]:
                return self._format_results(results, 0.85)

            return []

        except Exception as e:
            print(f"⚠️ Type search failed: {e}")
            return []

    def _search_by_title(self, query: str, k: int, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """Search for query keywords in section titles."""
        try:
            # Get all chunks and filter by title match
            q_words = set(query.lower().split())
            # Remove common words
            stop = {"what", "is", "the", "on", "in", "of", "a", "an", "show", "me", "find", "read", "about", "does", "say", "tell"}
            q_words = q_words - stop

            if not q_words:
                return []

            all_results = self.embedder.collection.get(
                limit=200, include=["documents", "metadatas"]
            )

            docs = []
            for i, meta in enumerate(all_results["metadatas"]):
                title = meta.get("section_title", "").lower()
                if any(w in title for w in q_words):
                    docs.append({
                        "text": all_results["documents"][i],
                        "metadata": meta,
                        "score": 0.8
                    })

            return docs[:k]

        except Exception as e:
            print(f"⚠️ Title search failed: {e}")
            return []

    def _scan_metadata(self, field: str, value: str, k: int) -> List[Dict]:
        """Scan all chunks for partial metadata match."""
        try:
            all_results = self.embedder.collection.get(
                limit=200, include=["documents", "metadatas"]
            )
            docs = []
            for i, meta in enumerate(all_results["metadatas"]):
                field_val = str(meta.get(field, ""))
                if field_val.startswith(value) or value in field_val:
                    docs.append({
                        "text": all_results["documents"][i],
                        "metadata": meta,
                        "score": 0.9
                    })
            return docs[:k]
        except:
            return []

    def _format_results(self, results: Dict, score: float) -> List[Dict]:
        """Format ChromaDB get() results into standard doc format."""
        docs = []
        for i in range(len(results["ids"])):
            docs.append({
                "text": results["documents"][i],
                "metadata": results["metadatas"][i],
                "score": score
            })
        return docs