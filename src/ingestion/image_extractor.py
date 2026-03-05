"""
Image and table extraction from PDFs.
Extracts meaningful figures/charts (filters out logos/icons)
and tables (markdown format for text embedding).
"""

import fitz  # PyMuPDF
import io
import uuid
from PIL import Image
from typing import List, Dict, Optional
from src.config import MIN_IMAGE_SIZE, MIN_IMAGE_BYTES


def extract_images(pdf_path: str, doc_id: Optional[str] = None) -> List[Dict]:
    """
    Extract meaningful images from a PDF.
    Filters out tiny icons, logos, and decorative elements.

    Args:
        pdf_path: Path to PDF file
        doc_id: Document ID for metadata

    Returns:
        List of dicts with 'id', 'image' (PIL), 'page', 'width', 'height'
    """
    if not doc_id:
        doc_id = str(uuid.uuid4())[:8]

    doc = fitz.open(pdf_path)
    images = []
    seen_hashes = set()

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]

            try:
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]

                # Filter: too small in bytes
                if len(img_bytes) < MIN_IMAGE_BYTES:
                    continue

                pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                w, h = pil_image.size

                # Filter: too small in pixels
                if w < MIN_IMAGE_SIZE or h < MIN_IMAGE_SIZE:
                    continue

                # Filter: duplicate detection via hash
                img_hash = hash(img_bytes[:1000])
                if img_hash in seen_hashes:
                    continue
                seen_hashes.add(img_hash)

                img_id = f"{doc_id}_img_p{page_num + 1}_{img_idx}"
                images.append({
                    "id": img_id,
                    "image": pil_image,
                    "page": page_num + 1,
                    "width": w,
                    "height": h,
                    "doc_id": doc_id,
                })

            except Exception:
                continue

    doc.close()
    return images


def extract_tables(pdf_path: str, doc_id: Optional[str] = None) -> List[Dict]:
    """
    Extract tables from PDF as markdown text.
    Tries camelot first, falls back to PyMuPDF.

    Args:
        pdf_path: Path to PDF file
        doc_id: Document ID for metadata

    Returns:
        List of dicts with 'id', 'markdown', 'page', 'rows', 'cols'
    """
    if not doc_id:
        doc_id = str(uuid.uuid4())[:8]

    # Try camelot first (better quality)
    tables = _extract_tables_camelot(pdf_path, doc_id)
    if tables:
        return tables

    # Fallback: PyMuPDF built-in
    return _extract_tables_pymupdf(pdf_path, doc_id)


def _extract_tables_camelot(pdf_path: str, doc_id: str) -> List[Dict]:
    try:
        import camelot

        raw_tables = camelot.read_pdf(pdf_path, pages="all", flavor="lattice")
        tables = []

        for i, table in enumerate(raw_tables):
            df = table.df
            if len(df) < 2:
                continue

            tables.append({
                "id": f"{doc_id}_table_{i}",
                "markdown": df.to_markdown(index=False),
                "page": table.page,
                "rows": len(df),
                "cols": len(df.columns),
                "doc_id": doc_id,
            })

        return tables

    except (ImportError, Exception):
        return []


def _extract_tables_pymupdf(pdf_path: str, doc_id: str) -> List[Dict]:
    try:
        doc = fitz.open(pdf_path)
        tables = []
        table_count = 0

        for page_num in range(len(doc)):
            page = doc[page_num]

            try:
                page_tables = page.find_tables()
            except Exception:
                continue

            for tab in page_tables:
                try:
                    data = tab.extract()
                    if not data or len(data) < 2:
                        continue

                    # Build markdown
                    header = "| " + " | ".join(str(c) for c in data[0]) + " |"
                    sep = "| " + " | ".join("---" for _ in data[0]) + " |"
                    rows = [
                        "| " + " | ".join(str(c) for c in row) + " |"
                        for row in data[1:]
                    ]
                    markdown = "\n".join([header, sep] + rows)

                    tables.append({
                        "id": f"{doc_id}_table_{table_count}",
                        "markdown": markdown,
                        "page": page_num + 1,
                        "rows": len(data),
                        "cols": len(data[0]),
                        "doc_id": doc_id,
                    })
                    table_count += 1

                except Exception:
                    continue

        doc.close()
        return tables

    except Exception:
        return []


def prepare_images_for_indexing(
    images: List[Dict],
    clip_embedder
) -> List[Dict]:
    """
    Embed extracted images with CLIP and prepare for ChromaDB indexing.

    Args:
        images: Output from extract_images()
        clip_embedder: CLIPEmbedder instance

    Returns:
        List of dicts ready for clip_embedder.index_images()
    """
    prepared = []

    # Batch embed for speed
    pil_images = [img["image"] for img in images]
    embeddings = clip_embedder.embed_images_batch(pil_images)

    for img, emb in zip(images, embeddings):
        prepared.append({
            "id": img["id"],
            "embedding": emb,
            "description": f"Image on page {img['page']} ({img['width']}x{img['height']})",
            "metadata": {
                "chunk_id": img["id"],
                "page_number": img["page"],
                "section_type": "image",
                "section_title": f"Figure on page {img['page']}",
                "section_number": str(img["page"]),
                "doc_id": img["doc_id"],
                "content_type": "image",
                "width": img["width"],
                "height": img["height"],
            }
        })

    return prepared


def prepare_tables_for_indexing(tables: List[Dict]) -> List[Dict]:
    """
    Prepare extracted tables for text collection indexing.
    Tables are text (markdown) → go into MiniLM collection, not CLIP.

    Returns:
        List of dicts ready for embedder.index_chunks()
    """
    return [
        {
            "id": t["id"],
            "text": t["markdown"],
            "metadata": {
                "chunk_id": t["id"],
                "page_number": t["page"],
                "section_type": "table",
                "section_title": f"Table on page {t['page']} ({t['rows']}x{t['cols']})",
                "section_number": str(t["page"]),
                "doc_id": t["doc_id"],
                "content_type": "table",
            }
        }
        for t in tables
    ]