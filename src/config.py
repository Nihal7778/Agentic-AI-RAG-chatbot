"""
Central configuration for Agentic RAG Chatbot.
All settings in one place — no magic strings scattered around.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# === Paths ===
ROOT_DIR = Path(__file__).parent.parent
USER_MEMORY_PATH = ROOT_DIR / "USER_MEMORY.md"
COMPANY_MEMORY_PATH = ROOT_DIR / "COMPANY_MEMORY.md"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
SAMPLE_DOCS_DIR = ROOT_DIR / "sample_docs"
CHROMA_PERSIST_DIR = str(ROOT_DIR / "chroma_db")

# === OpenAI ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.3          # Low for factual document analysis
HYDE_TEMPERATURE = 0.7          # Higher for diverse hypotheses
HYDE_NUM_HYPOTHESES = 2        # Reduced from 3 for speed

# === Embeddings ===
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384

# === ChromaDB ===
CHROMA_COLLECTION_NAME = "documents"

# === Chunking ===
CHUNK_SIZE = 512                # Max tokens per chunk
CHUNK_OVERLAP = 50              # Overlap between chunks
MIN_CHUNK_SIZE = 50             # Skip tiny chunks

# === Retrieval ===
TOP_K = 5                       # Number of docs to retrieve
RELEVANCE_THRESHOLD = 0.6       # Min score to consider relevant
MAX_RETRIES = 1                 # 1 retry for quality (set to 0 to disable)

# === Memory ===
MEMORY_CONFIDENCE_THRESHOLD = 0.7  # Only write when confidence > this
MAX_CONVERSATION_HISTORY = 10      # Keep last N turns for context

# === Section Types (for documents) ===
SECTION_TYPES = [
    "introduction", "background", "literature_review",
    "methodology", "results", "discussion",
    "conclusion", "references", "abstract",
    "related_work", "experimental_setup", "analysis",
    "future_work", "acknowledgements", "appendix",
    "general"
]

# === Complexity Levels (replaces risk scoring for docs) ===
COMPLEXITY_LEVELS = {
    "high": "🔴",
    "medium": "🟡",
    "low": "🟢"
}

HIGH_COMPLEXITY_SECTIONS = [
    "methodology", "experimental_setup", "results", "analysis"
]

MEDIUM_COMPLEXITY_SECTIONS = [
    "discussion", "literature_review", "related_work", "background"
]