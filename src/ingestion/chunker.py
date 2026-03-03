"""
Section-aware chunker for Agentic RAG Chatbot.
Splits document sections into chunks that respect boundaries.
Each chunk carries metadata for retrieval and citation.
"""

import uuid
import re
from typing import List, Dict
from dataclasses import dataclass
from .parser import ParsedDocument, Section
from ..config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE


@dataclass
class Chunk:
    """A single indexed chunk with metadata for retrieval + citation."""
    chunk_id: str
    text: str
    section_number: str
    section_title: str
    page_number: int
    section_type: str
    doc_id: str
    char_start: int
    char_end: int


def chunk_document(document: ParsedDocument, doc_id: str = None) -> List[Chunk]:
    """
    Split a parsed document into retrieval-ready chunks.

    Strategy:
    - Short sections (< CHUNK_SIZE) → keep as single chunk
    - Long sections → split at sentence boundaries with overlap
    - Each chunk inherits section metadata for filtering + citations

    Args:
        document: Parsed document with sections
        doc_id: Unique ID for this document (auto-generated if None)

    Returns:
        List of Chunk objects ready for embedding
    """
    if not doc_id:
        doc_id = str(uuid.uuid4())[:8]

    chunks = []

    for section in document.sections:
        text = section.text.strip()

        if len(text) < MIN_CHUNK_SIZE:
            continue

        if _estimate_tokens(text) <= CHUNK_SIZE:
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_s{section.section_number}_{len(chunks)}",
                text=text,
                section_number=section.section_number,
                section_title=section.title,
                page_number=section.page_number,
                section_type=section.section_type,
                doc_id=doc_id,
                char_start=section.start_index,
                char_end=section.start_index + len(text)
            ))
        else:
            sub_chunks = _split_at_sentences(text, CHUNK_SIZE, CHUNK_OVERLAP)
            for sub_text in sub_chunks:
                if len(sub_text.strip()) < MIN_CHUNK_SIZE:
                    continue
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}_s{section.section_number}_{len(chunks)}",
                    text=sub_text.strip(),
                    section_number=section.section_number,
                    section_title=section.title,
                    page_number=section.page_number,
                    section_type=section.section_type,
                    doc_id=doc_id,
                    char_start=section.start_index,
                    char_end=section.start_index + len(sub_text)
                ))

    return chunks


def chunks_to_documents(chunks: List[Chunk]) -> List[Dict]:
    """Convert chunks to format expected by ChromaDB."""
    docs = []
    for c in chunks:
        docs.append({
            "id": c.chunk_id,
            "text": c.text,
            "metadata": {
                "chunk_id": c.chunk_id,
                "section_number": c.section_number,
                "section_title": c.section_title,
                "page_number": c.page_number,
                "section_type": c.section_type,
                "doc_id": c.doc_id,
            }
        })
    return docs


# === Helpers ===

def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def _split_at_sentences(text: str, max_tokens: int, overlap: int) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = _estimate_tokens(sent)

        if current_tokens + sent_tokens > max_tokens and current:
            chunks.append(" ".join(current))
            overlap_sents = []
            overlap_tokens = 0
            for s in reversed(current):
                t = _estimate_tokens(s)
                if overlap_tokens + t > overlap:
                    break
                overlap_sents.insert(0, s)
                overlap_tokens += t
            current = overlap_sents
            current_tokens = overlap_tokens

        current.append(sent)
        current_tokens += sent_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks