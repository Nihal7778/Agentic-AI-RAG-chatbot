"""
Multi-HyDE RAG retrieval for complex queries.
Generates hypothetical academic/research passages to bridge
the gap between user questions and technical document language.
"""

import time
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.retrieval.embedder import ContractEmbedder
from src.config import LLM_MODEL, HYDE_TEMPERATURE, HYDE_NUM_HYPOTHESES, TOP_K


GENERAL_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=(
        "You are a research document analyst. Given a user's question "
        "about an academic or technical document, write a short passage (2-3 sentences) "
        "that would typically appear in a research paper or textbook addressing this topic.\n\n"
        "Use formal academic language, technical terminology, and cite-style references "
        "where appropriate.\n\n"
        "Question: {question}\n\nHypothetical academic passage:"
    ),
)

TECHNICAL_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=(
        "You are a machine learning and data science expert. Given a "
        "technical question, write a short passage (2-3 sentences) that would appear "
        "in an ML research paper or textbook explaining this concept.\n\n"
        "Include specific technical terms like model architectures, algorithms, "
        "evaluation metrics, training procedures, and mathematical concepts.\n\n"
        "Question: {question}\n\nHypothetical technical passage:"
    ),
)


# ── Cached chains (avoid re-creating LLM client per call) ───

_chain_cache: Dict[bool, any] = {}


def _get_hyde_chain(is_technical: bool = False):
    if is_technical not in _chain_cache:
        llm = ChatOpenAI(model=LLM_MODEL, temperature=HYDE_TEMPERATURE, max_tokens=300)
        prompt = TECHNICAL_PROMPT if is_technical else GENERAL_PROMPT
        _chain_cache[is_technical] = prompt | llm | StrOutputParser()
    return _chain_cache[is_technical]


# ── Helpers ──────────────────────────────────────────────────

def _generate_hypotheses(
    query: str, n: int = HYDE_NUM_HYPOTHESES, is_technical: bool = False
) -> List[str]:
    """Generate N hypothetical passages in a single batch call."""
    chain = _get_hyde_chain(is_technical)
    inputs = [{"question": query}] * n
    try:
        results = chain.batch(inputs)
        return [h.strip() for h in results if h and h.strip()]
    except Exception:
        return []


def _merge_and_dedup(docs: List[Dict], k: int) -> List[Dict]:
    seen = {}
    for doc in docs:
        cid = doc["metadata"].get("chunk_id", id(doc))
        if cid not in seen or doc["score"] > seen[cid]["score"]:
            seen[cid] = doc
    return sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:k]


def retrieve_multi_hyde(
    embedder: ContractEmbedder,
    query: str,
    k: int = TOP_K,
    filter_dict: Optional[Dict] = None,
    is_technical: bool = False,
) -> tuple[List[Dict], List[str]]:
    """
    Core Multi-HyDE pipeline (optimized):
    1. Batch-generate N hypothetical passages (parallel LLM calls)
    2. Batch-embed all hypotheses + original query (single embedding call)
    3. Search ChromaDB with each embedding
    4. Merge and deduplicate
    Returns (docs, hypotheses).
    """
    hypotheses = _generate_hypotheses(query, HYDE_NUM_HYPOTHESES, is_technical)

    texts_to_embed = [query] + hypotheses
    all_embeddings = embedder.embed_batch(texts_to_embed)

    all_docs = []
    for emb in all_embeddings:
        docs = embedder.search_by_embedding(emb, k=k, filter_dict=filter_dict)
        all_docs.extend(docs)

    return _merge_and_dedup(all_docs, k), hypotheses


# ── Class wrapper (backward-compatible with orchestrator) ────

class MultiHyDERetriever:
    """Drop-in replacement preserving the original interface."""

    def __init__(self, embedder: ContractEmbedder):
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        k: int = TOP_K,
        filter_dict: Optional[Dict] = None,
        is_technical_query: bool = False,
    ) -> Dict:
        start = time.time()
        docs, hypotheses = retrieve_multi_hyde(
            self.embedder, query, k, filter_dict, is_technical_query
        )
        return {
            "documents": docs,
            "hypotheses": hypotheses,
            "retrieval_time": time.time() - start,
            "strategy": "multi_hyde",
        }