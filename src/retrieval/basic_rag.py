"""
Basic RAG retrieval using LangChain's dynamic_prompt middleware.
Tries metadata filters first (section number, section type),
falls back to semantic search.
"""

import re
from typing import Optional, Dict
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from src.retrieval.embedder import ContractEmbedder
from src.config import TOP_K


SECTION_TYPE_KEYWORDS = {
    "introduction": ["introduction", "intro", "overview"],
    "methodology": ["methodology", "methods", "approach"],
    "results": ["results", "findings", "performance"],
    "discussion": ["discussion", "implications"],
    "conclusion": ["conclusion", "summary", "concluding"],
    "references": ["references", "bibliography"],
    "related_work": ["related work", "literature"],
    "background": ["background", "preliminaries"],
    "abstract": ["abstract"],
}


# ── Helpers ──────────────────────────────────────────────────

def _extract_section_number(query: str) -> Optional[str]:
    for p in [r"chapter\s*(\d+)", r"section\s*(\d+(?:\.\d+)*)", r"page\s*(\d+)"]:
        m = re.search(p, query.lower())
        if m:
            return m.group(1)
    return None


def _extract_section_type(query: str) -> Optional[str]:
    q = query.lower()
    for sec_type, kws in SECTION_TYPE_KEYWORDS.items():
        if any(kw in q for kw in kws):
            return sec_type
    return None


def _metadata_search(
    collection, where: Dict, k: int = TOP_K
) -> list[dict]:
    """Run a ChromaDB metadata get() and return formatted docs."""
    results = collection.get(where=where, limit=k, include=["documents", "metadatas"])
    if not results["ids"]:
        return []
    return [
        {"text": results["documents"][i], "metadata": results["metadatas"][i], "score": 0.95}
        for i in range(len(results["ids"]))
    ]


# ── Core retrieval logic ─────────────────────────────────────

def retrieve_contract_context(
    embedder: ContractEmbedder,
    query: str,
    k: int = TOP_K,
    filter_dict: Optional[Dict] = None,
) -> str:
    """
    Smart retrieval: metadata filters first, semantic fallback.
    Returns serialized context string for prompt injection.
    """
    docs = []

    # 1) Section number lookup
    sec_num = _extract_section_number(query)
    if sec_num:
        where = {"section_number": sec_num}
        if filter_dict:
            where = {"$and": [where, filter_dict]}
        docs = _metadata_search(embedder.collection, where, k)

    # 2) Section type lookup
    if not docs:
        sec_type = _extract_section_type(query)
        if sec_type:
            where = {"section_type": sec_type}
            if filter_dict:
                where = {"$and": [where, filter_dict]}
            docs = _metadata_search(embedder.collection, where, k)

    # 3) Semantic fallback
    if not docs:
        raw = embedder.search(query=query, k=k, filter_dict=filter_dict)
        docs = raw if raw else []

    return "\n\n".join(
        f"[Source: {d.get('metadata', {})}]\n{d.get('text', '')}" for d in docs
    )


# ── Class wrapper (backward-compatible with orchestrator) ────

class BasicRAGRetriever:
    """Drop-in replacement preserving the original interface."""

    def __init__(self, embedder: ContractEmbedder):
        self.embedder = embedder

    def retrieve(self, query: str, k: int = TOP_K, filter_dict: Optional[Dict] = None) -> Dict:
        import time
        start = time.time()
        docs = []

        sec_num = _extract_section_number(query)
        if sec_num:
            where = {"section_number": sec_num}
            if filter_dict:
                where = {"$and": [where, filter_dict]}
            docs = _metadata_search(self.embedder.collection, where, k)
            if docs:
                return {"documents": docs, "retrieval_time": time.time() - start, "strategy": "basic_section_number"}

        sec_type = _extract_section_type(query)
        if sec_type:
            where = {"section_type": sec_type}
            if filter_dict:
                where = {"$and": [where, filter_dict]}
            docs = _metadata_search(self.embedder.collection, where, k)
            if docs:
                return {"documents": docs, "retrieval_time": time.time() - start, "strategy": "basic_section_type"}

        docs = self.embedder.search(query=query, k=k, filter_dict=filter_dict) or []
        return {"documents": docs, "retrieval_time": time.time() - start, "strategy": "basic_semantic"}


# ── Option A: dynamic_prompt middleware (RAG chain) ──────────
# Single LLM call — always retrieves then generates.

def create_basic_rag_middleware(embedder: ContractEmbedder, k: int = TOP_K):
    """Factory that returns a dynamic_prompt wired to your embedder."""

    @dynamic_prompt
    def inject_contract_context(request: ModelRequest) -> str:
        query = request.state["messages"][-1].text
        context = retrieve_contract_context(embedder, query, k)
        return (
            "You are a research document analyst. Given a user's question "
            "about an academic or technical document, write a short passage "
            "(2-3 sentences) that would typically appear in a research paper "
            "or textbook addressing this topic. Use formal academic language, "
            "technical terminology, and cite-style references where appropriate.\n\n"
            "You are also a machine learning and data science expert. For "
            "technical questions, include specific technical terms like model "
            "architectures, algorithms, evaluation metrics, training procedures, "
            "and mathematical concepts.\n\n"
            "Use the following retrieved context to answer:\n\n"
            f"{context}"
        )

    return inject_contract_context


def build_basic_rag_agent(model, embedder: ContractEmbedder, k: int = TOP_K):
    """Build a single-call RAG agent using the middleware pattern."""
    middleware = create_basic_rag_middleware(embedder, k)
    return create_agent(model, tools=[], middleware=[middleware])


# ── Option B: @tool-based (agentic RAG) ─────────────────────
# Two LLM calls — model decides when to search.

def create_retrieval_tool(embedder: ContractEmbedder, k: int = TOP_K):
    """Create a LangChain tool wrapping your contract retriever."""

    @tool(response_format="content_and_artifact")
    def retrieve_documents(query: str):
        """Retrieve relevant sections from research or technical documents."""
        context = retrieve_contract_context(embedder, query, k)
        return context, context

    return retrieve_documents


def build_agentic_rag(model, embedder: ContractEmbedder, k: int = TOP_K):
    """Build an agentic RAG where the LLM decides when to retrieve."""
    tools = [create_retrieval_tool(embedder, k)]
    prompt = (
        "You are a research document analyst and machine learning expert. "
        "Use the retrieve_documents tool to find relevant sections from "
        "academic or technical documents before answering. Write responses "
        "using formal academic language, technical terminology, and "
        "cite-style references. Include specific terms like model "
        "architectures, algorithms, evaluation metrics, and mathematical concepts."
    )
    return create_agent(model, tools, system_prompt=prompt)