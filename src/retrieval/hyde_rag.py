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
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
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
    # Step 1: Batch hypothesis generation
    hypotheses = _generate_hypotheses(query, HYDE_NUM_HYPOTHESES, is_technical)

    # Step 2: Batch embed — original query + all hypotheses in one call
    texts_to_embed = [query] + hypotheses
    all_embeddings = embedder.embed_batch(texts_to_embed)

    # Step 3: Search with each embedding
    all_docs = []
    for emb in all_embeddings:
        docs = embedder.search_by_embedding(emb, k=k, filter_dict=filter_dict)
        all_docs.extend(docs)

    # Step 4: Merge and deduplicate
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


# ── Option A: dynamic_prompt middleware (RAG chain) ──────────

def create_multi_hyde_middleware(
    embedder: ContractEmbedder, k: int = TOP_K, is_technical: bool = False
):
    @dynamic_prompt
    def inject_hyde_context(request: ModelRequest) -> str:
        query = request.state["messages"][-1].text
        docs, _ = retrieve_multi_hyde(embedder, query, k, is_technical=is_technical)
        context = "\n\n".join(
            f"[Source: {d.get('metadata', {})}]\n{d.get('text', '')}" for d in docs
        )
        return (
            "You are a research document analyst and machine learning expert. "
            "Given a user's question about an academic or technical document, "
            "write a response using formal academic language, technical terminology, "
            "and cite-style references where appropriate. "
            "Include specific technical terms like model architectures, algorithms, "
            "evaluation metrics, training procedures, and mathematical concepts.\n\n"
            "Use the following retrieved context to answer:\n\n"
            f"{context}"
        )

    return inject_hyde_context


def build_multi_hyde_agent(
    model, embedder: ContractEmbedder, k: int = TOP_K, is_technical: bool = False
):
    middleware = create_multi_hyde_middleware(embedder, k, is_technical)
    return create_agent(model, tools=[], middleware=[middleware])


# ── Option B: @tool-based (agentic RAG) ─────────────────────

def create_hyde_retrieval_tool(
    embedder: ContractEmbedder, k: int = TOP_K, is_technical: bool = False
):
    @tool(response_format="content_and_artifact")
    def retrieve_with_hyde(query: str):
        """Retrieve relevant sections using hypothetical document embeddings for deeper semantic matching."""
        docs, hypotheses = retrieve_multi_hyde(embedder, query, k, is_technical=is_technical)
        context = "\n\n".join(
            f"[Source: {d.get('metadata', {})}]\n{d.get('text', '')}" for d in docs
        )
        return context, {"documents": docs, "hypotheses": hypotheses}

    return retrieve_with_hyde


def build_agentic_hyde(
    model, embedder: ContractEmbedder, k: int = TOP_K, is_technical: bool = False
):
    tools = [create_hyde_retrieval_tool(embedder, k, is_technical)]
    prompt = (
        "You are a research document analyst and machine learning expert. "
        "Use the retrieve_with_hyde tool for complex queries that need "
        "deeper semantic matching. Write responses using formal academic "
        "language and technical terminology."
    )
    return create_agent(model, tools, system_prompt=prompt)