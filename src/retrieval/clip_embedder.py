"""
CLIP Embedder for multimodal RAG.
Handles image and cross-modal text embeddings using CLIP ViT-B/32.
Separate from ContractEmbedder — different model, different dimensions.
"""

import torch
import chromadb
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
from PIL import Image
from typing import List, Dict, Optional
from src.config import (
    CLIP_MODEL_NAME, CHROMA_PERSIST_DIR,
    CLIP_COLLECTION_NAME, TOP_K, IMAGE_RELEVANCE_THRESHOLD
)


class CLIPEmbedder:
    """Manages CLIP embeddings and image ChromaDB collection."""

    def __init__(self):
        self.model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        self.tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_NAME)
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection = self.client.get_or_create_collection(
            name=CLIP_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def _to_tensor(self, output) -> torch.Tensor:
        """Handle both raw tensor and model output object formats."""
        if isinstance(output, torch.Tensor):
            return output
        if hasattr(output, "pooler_output"):
            return output.pooler_output
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state[:, 0, :]
        raise ValueError(f"Unexpected CLIP output type: {type(output)}")

    def embed_text(self, text: str) -> List[float]:
        """Embed text into CLIP's shared vector space."""
        inputs = self.tokenizer(
            text, return_tensors="pt",
            padding=True, truncation=True, max_length=77
        )
        with torch.no_grad():
            raw = self.model.get_text_features(**inputs)
        emb = self._to_tensor(raw)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        return emb[0].cpu().numpy().tolist()

    def embed_image(self, pil_image: Image.Image) -> List[float]:
        """Embed a PIL image into CLIP's shared vector space."""
        inputs = self.processor(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            raw = self.model.get_image_features(**inputs)
        emb = self._to_tensor(raw)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        return emb[0].cpu().numpy().tolist()

    def embed_images_batch(self, images: List[Image.Image]) -> List[List[float]]:
        """Batch embed multiple PIL images."""
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        with torch.no_grad():
            raw = self.model.get_image_features(**inputs)
        emb = self._to_tensor(raw)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        return emb.cpu().numpy().tolist()

    def index_images(self, images: List[Dict]) -> int:
        """
        Store image embeddings in ChromaDB.

        Args:
            images: List of dicts with 'id', 'embedding', 'metadata', 'description'

        Returns:
            Number of images indexed
        """
        if not images:
            return 0

        self.collection.add(
            ids=[img["id"] for img in images],
            embeddings=[img["embedding"] for img in images],
            metadatas=[img["metadata"] for img in images],
            documents=[img["description"] for img in images],
        )
        return len(images)

    def search(
        self,
        query: str,
        k: int = TOP_K,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """Search image collection using CLIP text embedding."""
        query_emb = self.embed_text(query)

        kwargs = {
            "query_embeddings": [query_emb],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"]
        }
        if filter_dict:
            kwargs["where"] = filter_dict

        results = self.collection.query(**kwargs)

        docs = []
        for i in range(len(results["ids"][0])):
            score = 1 - results["distances"][0][i]
            if score >= IMAGE_RELEVANCE_THRESHOLD:
                docs.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": score
                })
        return docs

    def get_collection_count(self) -> int:
        return self.collection.count()

    def reset(self):
        self.client.delete_collection(CLIP_COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=CLIP_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )