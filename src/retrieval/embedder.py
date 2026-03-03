"""
Embedding + ChromaDB indexing for Contract Sidekick.
Handles storing and searching contract chunks.
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from src.config import (
    EMBEDDING_MODEL, CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME, TOP_K
)


class ContractEmbedder:
    """Manages embeddings and ChromaDB vector store."""

    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        return self.model.encode(text).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts at once (faster)."""
        return self.model.encode(texts).tolist()

    def index_chunks(self, chunks: List[Dict]) -> int:
        """
        Store chunks in ChromaDB with embeddings + metadata.
        
        Args:
            chunks: List of dicts with 'id', 'text', 'metadata'
            
        Returns:
            Number of chunks indexed
        """
        if not chunks:
            return 0

        ids = [c["id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        embeddings = self.embed_batch(texts)

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

        return len(chunks)

    def search(
        self,
        query: str,
        k: int = TOP_K,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Similarity search using query embedding.
        
        Args:
            query: Search query text
            k: Number of results
            filter_dict: Optional metadata filter e.g. {"clause_type": "indemnification"}
            
        Returns:
            List of {text, metadata, score}
        """
        query_embedding = self.embed_text(query)

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"]
        }
        if filter_dict:
            kwargs["where"] = filter_dict

        results = self.collection.query(**kwargs)

        docs = []
        for i in range(len(results["ids"][0])):
            docs.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i]  # Convert distance to similarity
            })

        return docs

    def search_by_embedding(
        self,
        embedding: List[float],
        k: int = TOP_K,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search using a pre-computed embedding (used by HyDE).
        
        Args:
            embedding: Pre-computed embedding vector
            k: Number of results
            filter_dict: Optional metadata filter
            
        Returns:
            List of {text, metadata, score}
        """
        kwargs = {
            "query_embeddings": [embedding],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"]
        }
        if filter_dict:
            kwargs["where"] = filter_dict

        results = self.collection.query(**kwargs)

        docs = []
        for i in range(len(results["ids"][0])):
            docs.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i]
            })

        return docs

    def delete_document(self, doc_id: str):
        """Delete all chunks belonging to a document."""
        self.collection.delete(where={"doc_id": doc_id})

    def get_collection_count(self) -> int:
        """Return total number of chunks in the store."""
        return self.collection.count()

    def reset(self):
        """Clear the entire collection."""
        self.client.delete_collection(CHROMA_COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )