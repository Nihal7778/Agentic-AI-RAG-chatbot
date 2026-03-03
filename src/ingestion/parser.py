"""
PDF Parser for Agentic RAG Chatbot.
Extracts text from PDFs and detects section boundaries.
Handles academic papers, textbooks, and research documents.
Uses PyMuPDF (fitz) for reliable extraction.
"""

import fitz  # PyMuPDF
import re
from typing import List, Dict
from dataclasses import dataclass, field


@dataclass
class Section:
    """Represents a detected section in the document."""
    section_number: str          # e.g. "5.2", "Chapter 3"
    title: str                   # e.g. "Methodology"
    text: str                    # Full text of the section
    page_number: int             # Page where section starts
    start_index: int             # Character index in full text
    section_type: str = "general"  # Classified type


@dataclass
class ParsedDocument:
    """Result of parsing a PDF document."""
    full_text: str
    sections: List[Section]
    metadata: Dict = field(default_factory=dict)
    page_count: int = 0
    title: str = ""


# Common section header patterns in  documents
SECTION_PATTERNS = [
    # "Chapter 1" or "CHAPTER 1"
    r'(?:^|\n)\s*((?:Chapter|CHAPTER)\s+\d+(?:\.\d+)*\.?\s*[:\-\.]?\s*.+)',
    # "Section 1.2" or "SECTION 1.2"
    r'(?:^|\n)\s*((?:Section|SECTION)\s+\d+(?:\.\d+)*\.?\s*[:\-\.]?\s*.+)',
    # "1.2 Title" or "1.2. Title" (numbered sections)
    r'(?:^|\n)\s*(\d+\.\d+(?:\.\d+)*\.?\s+[A-Z][^\n]{3,80})',
    # "1. Title" (top-level numbered)
    r'(?:^|\n)\s*(\d+\.\s+[A-Z][^\n]{3,80})',
    # "Abstract", "Introduction", "Conclusion", etc. (standalone headers)
    r'(?:^|\n)\s*((?:Abstract|Introduction|Background|Methodology|Methods|Results|Discussion|Conclusion|Conclusions|References|Bibliography|Acknowledgements|Appendix|Related Work|Literature Review|Future Work|Experimental Setup)\s*\n)',
    # ALL CAPS headers
    r'(?:^|\n)\s*([A-Z][A-Z\s]{4,60}(?:\n|$))',
]

# Section type keywords for classification
SECTION_KEYWORDS = {
    "abstract": ["abstract", "summary of the paper"],
    "introduction": ["introduction", "overview", "motivation", "background and motivation"],
    "background": ["background", "preliminaries", "foundations", "theoretical framework"],
    "literature_review": ["literature review", "survey", "state of the art", "prior work"],
    "related_work": ["related work", "related studies", "previous work", "existing approaches"],
    "methodology": ["methodology", "methods", "approach", "proposed method", "proposed approach", "framework", "system design", "model design"],
    "experimental_setup": ["experimental setup", "experiment", "implementation", "setup", "dataset", "data collection", "data acquisition"],
    "results": ["results", "findings", "performance", "evaluation", "performance results"],
    "analysis": ["analysis", "discussion of results", "comparative analysis", "ablation"],
    "discussion": ["discussion", "implications", "interpretation", "limitations"],
    "conclusion": ["conclusion", "concluding remarks", "summary", "future work and conclusion"],
    "future_work": ["future work", "future directions", "open problems", "future scope"],
    "references": ["references", "bibliography", "works cited"],
    "acknowledgements": ["acknowledgement", "acknowledgment", "funding"],
    "appendix": ["appendix", "supplementary", "additional material"],
}


def extract_text_from_pdf(pdf_path: str) -> ParsedDocument:
    """
    Extract text from a PDF file with page tracking.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        ParsedDocument with full text and metadata
    """
    doc = fitz.open(pdf_path)
    full_text = ""
    page_texts = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        page_texts.append({"page": page_num + 1, "text": text, "start": len(full_text)})
        full_text += text + "\n"

    title = _extract_title(page_texts[0]["text"] if page_texts else "")
    doc.close()

    return ParsedDocument(
        full_text=full_text.strip(),
        sections=[],
        metadata={"source": pdf_path, "page_texts": page_texts},
        page_count=len(page_texts),
        title=title
    )


def detect_sections(document: ParsedDocument) -> ParsedDocument:
    """
    Detect section boundaries in the document text.
    Updates document.sections in place.
    """
    text = document.full_text
    matches = []

    for pattern in SECTION_PATTERNS:
        for m in re.finditer(pattern, text):
            header = m.group(1).strip()
            if 3 < len(header) < 120:
                matches.append({
                    "header": header,
                    "start": m.start(1),
                    "page": _get_page_number(m.start(1), document.metadata.get("page_texts", []))
                })

    matches.sort(key=lambda x: x["start"])
    matches = _deduplicate_matches(matches)

    sections = []
    for i, match in enumerate(matches):
        end = matches[i + 1]["start"] if i + 1 < len(matches) else len(text)
        section_text = text[match["start"]:end].strip()
        sec_num = _extract_section_number(match["header"])
        section_type = classify_section(match["header"], section_text)

        sections.append(Section(
            section_number=sec_num,
            title=match["header"][:100],
            text=section_text,
            page_number=match["page"],
            start_index=match["start"],
            section_type=section_type
        ))

    document.sections = sections
    return document


def classify_section(title: str, text: str) -> str:
    """
    Classify a section by its title and content.

    Args:
        title: The section title/header
        text: The section body text

    Returns:
        Section type string (e.g. "methodology", "results")
    """
    combined = (title + " " + text[:500]).lower()

    best_match = "general"
    best_score = 0

    for sec_type, keywords in SECTION_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in combined)
        # Give extra weight if keyword appears in the title
        title_score = sum(2 for kw in keywords if kw in title.lower())
        total = score + title_score
        if total > best_score:
            best_score = total
            best_match = sec_type

    return best_match


def parse_document(pdf_path: str) -> ParsedDocument:
    """
    Full pipeline: extract text → detect sections → classify.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Fully parsed document with sections and classifications
    """
    document = extract_text_from_pdf(pdf_path)
    document = detect_sections(document)

    if not document.sections:
        document.sections = [Section(
            section_number="1",
            title=document.title or "Full Document",
            text=document.full_text,
            page_number=1,
            start_index=0,
            section_type=classify_section(document.title, document.full_text)
        )]

    return document


# === Helper Functions ===

def _extract_title(first_page_text: str) -> str:
    """Extract document title from first page."""
    lines = first_page_text.strip().split("\n")
    for line in lines[:10]:
        cleaned = line.strip()
        if len(cleaned) > 5 and len(cleaned) < 150:
            return cleaned
    return "Untitled Document"


def _get_page_number(char_index: int, page_texts: List[Dict]) -> int:
    """Find which page a character index falls on."""
    for pt in reversed(page_texts):
        if char_index >= pt["start"]:
            return pt["page"]
    return 1


def _extract_section_number(header: str) -> str:
    """Pull section/chapter number from header text."""
    m = re.match(r'(?:Chapter|CHAPTER|Section|SECTION)\s+(\d+(?:\.\d+)*)', header)
    if m:
        return m.group(1)

    m = re.match(r'(\d+(?:\.\d+)*)', header)
    if m:
        return m.group(1)

    return "?"


def _deduplicate_matches(matches: List[Dict], min_gap: int = 20) -> List[Dict]:
    """Remove overlapping section header matches."""
    if not matches:
        return []

    deduped = [matches[0]]
    for m in matches[1:]:
        if m["start"] - deduped[-1]["start"] > min_gap:
            deduped.append(m)
    return deduped