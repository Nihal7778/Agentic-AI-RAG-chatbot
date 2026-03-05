"""
Multimodal RAG retrieval.
Searches both text (MiniLM) and image (CLIP) collections,
merges results, and optionally describes images via GPT-4o vision.
"""

import io
import base64
from typing import List, Dict, Optional
from PIL import Image
from openai import OpenAI
from src.retrieval.embedder import ContractEmbedder
from src.retrieval.clip_embedder import CLIPEmbedder
from src.config import (
    TOP_K, IMAGE_RELEVANCE_THRESHOLD,
    VISION_MODEL, OPENAI_API_KEY
)


class MultimodalRetriever:
    """
    Dual-pipeline retrieval:
      Text query → MiniLM → text collection → text chunks
      Same query → CLIP  → image collection → image results
      Merge → flag has_images → optionally describe with GPT-4o
    """

    def __init__(
        self,
        text_embedder: ContractEmbedder,
        clip_embedder: CLIPEmbedder
    ):
        self.text_embedder = text_embedder
        self.clip_embedder = clip_embedder
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)

    def search(
        self,
        query: str,
        k_text: int = TOP_K,
        k_images: int = 3,
        filter_dict: Optional[Dict] = None,
    ) -> Dict:
        """
        Search both collections and merge results.

        Returns:
            {
                "text": list of text docs,
                "images": list of image docs,
                "has_images": bool,
            }
        """
        # Text search (MiniLM — existing pipeline)
        text_results = self.text_embedder.search(
            query=query, k=k_text, filter_dict=filter_dict
        )

        # Image search (CLIP — cross-modal)
        image_results = []
        try:
            image_results = self.clip_embedder.search(
                query=query, k=k_images, filter_dict=filter_dict
            )
        except Exception:
            pass  # Image search failure doesn't break text RAG

        has_images = len(image_results) > 0

        return {
            "text": text_results,
            "images": image_results,
            "has_images": has_images,
        }

    def describe_image(
        self,
        pil_image: Image.Image,
        query: str
    ) -> str:
        """
        Send an image to GPT-4o vision and get a description.

        Args:
            pil_image: PIL Image object
            query: User's question for context

        Returns:
            Text description of the image
        """
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        try:
            response = self.openai_client.chat.completions.create(
                model=VISION_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Based on this image from a research document, "
                                f"answer: {query}\n"
                                f"Be specific with any numbers or labels visible."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64}"
                            },
                        },
                    ],
                }],
                max_tokens=500,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[Image description unavailable: {e}]"

    def retrieve_with_descriptions(
        self,
        query: str,
        extracted_images: List[Dict],
        k_text: int = TOP_K,
        k_images: int = 2,
        filter_dict: Optional[Dict] = None,
    ) -> Dict:
        """
        Full multimodal retrieval with GPT-4o image descriptions.

        Args:
            query: User's question
            extracted_images: List of dicts with 'id' and 'image' (PIL)
                from image_extractor.extract_images()
            k_text: Number of text results
            k_images: Number of image results
            filter_dict: Optional metadata filter

        Returns:
            {
                "text": text docs,
                "images": image docs,
                "image_descriptions": list of description strings,
                "has_images": bool,
                "strategy": "multimodal" | "text_only"
            }
        """
        results = self.search(query, k_text, k_images, filter_dict)

        image_descriptions = []

        if results["has_images"]:
            # Build lookup: image_id → PIL image
            img_lookup = {img["id"]: img["image"] for img in extracted_images}

            for img_result in results["images"][:k_images]:
                img_id = img_result.get("id")
                pil_img = img_lookup.get(img_id)

                if pil_img:
                    desc = self.describe_image(pil_img, query)
                    page = img_result["metadata"].get("page_number", "?")
                    image_descriptions.append(
                        f"[Image, Page {page}]: {desc}"
                    )

        return {
            "text": results["text"],
            "images": results["images"],
            "image_descriptions": image_descriptions,
            "has_images": results["has_images"],
            "strategy": "multimodal" if results["has_images"] else "text_only",
        }