"""
Multi-HyDE RAG retrieval for complex queries.
Adapted from Nihal's existing HyDE implementation.

Generates hypothetical academic/research passages to bridge
the gap between user questions and technical document language.
"""

import time
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.retrieval.embedder import ContractEmbedder
from src.config import (
    LLM_MODEL, HYDE_TEMPERATURE, HYDE_NUM_HYPOTHESES, TOP_K
)


# Academic/research hypothesis prompts
GENERAL_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are a research document analyst. Given a user's question
about an academic or technical document, write a short passage (2-3 sentences)
that would typically appear in a research paper or textbook addressing this topic.

Use formal academic language, technical terminology, and cite-style references
where appropriate.

Question: {question}

Hypothetical academic passage:"""
)

TECHNICAL_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are a machine learning and data science expert. Given a
technical question, write a short passage (2-3 sentences) that would appear
in an ML research paper or textbook explaining this concept.

Include specific technical terms like model architectures, algorithms,
evaluation metrics, training procedures, and mathematical concepts.

Question: {question}

Hypothetical technical passage:"""
)


class MultiHyDERetriever:
    """
    Multi-hypothesis HyDE retriever for complex document queries.
    Generates N hypothetical passages, embeds each,
    searches with all + original query, merges and deduplicates.
    """

    def __init__(self, embedder):
        self.embedder = embedder
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=HYDE_TEMPERATURE,
            max_tokens=300
        )
        self.general_chain = GENERAL_PROMPT | self.llm | StrOutputParser()
        self.technical_chain = TECHNICAL_PROMPT | self.llm | StrOutputParser()

    def generate_hypotheses(
        self,
        query: str,
        n: int = HYDE_NUM_HYPOTHESES,
        is_technical_query: bool = False
    ) -> List[str]:
        """Generate N hypothetical document passages."""
        chain = self.technical_chain if is_technical_query else self.general_chain
        hypotheses = []

        for _ in range(n):
            try:
                h = chain.invoke({"question": query}).strip()
                if h:
                    hypotheses.append(h)
            except Exception as e:
                print(f"⚠️ HyDE generation failed: {e}")

        return hypotheses

    def retrieve(
        self,
        query: str,
        k: int = TOP_K,
        filter_dict: Optional[Dict] = None,
        is_technical_query: bool = False
    ) -> Dict:
        """
        Multi-HyDE retrieval pipeline.

        1. Generate N hypothetical passages
        2. Embed each hypothesis + original query
        3. Search ChromaDB with ALL embeddings
        4. Merge results, deduplicate by chunk_id
        5. Return top-k unique results
        """
        # Step 1: Generate hypotheses
        hyde_start = time.time()
        hypotheses = self.generate_hypotheses(query, HYDE_NUM_HYPOTHESES, is_technical_query)
        hyde_time = time.time() - hyde_start

        retrieval_start = time.time()

        # Step 2: Search with original query
        all_docs = self.embedder.search(query, k=k, filter_dict=filter_dict)

        # Step 3: Search with each hypothesis
        for hyp in hypotheses:
            hyp_embedding = self.embedder.embed_text(hyp)
            hyp_docs = self.embedder.search_by_embedding(
                hyp_embedding, k=k, filter_dict=filter_dict
            )
            all_docs.extend(hyp_docs)

        # Step 4: Merge and deduplicate
        merged = self._merge_and_dedup(all_docs, k)

        return {
            "documents": merged,
            "hypotheses": hypotheses,
            "retrieval_time": time.time() - retrieval_start,
            "hyde_generation_time": hyde_time,
            "strategy": "multi_hyde"
        }

    def _merge_and_dedup(self, docs: List[Dict], k: int) -> List[Dict]:
        """Deduplicate by chunk_id, keep highest scoring version."""
        seen = {}
        for doc in docs:
            cid = doc["metadata"].get("chunk_id", id(doc))
            if cid not in seen or doc["score"] > seen[cid]["score"]:
                seen[cid] = doc

        ranked = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
        return ranked[:k]