"""
Memory Reader for Agentic RAG Chatbot.
Reads USER_MEMORY.md and COMPANY_MEMORY.md to provide context.
"""

from src.config import USER_MEMORY_PATH, COMPANY_MEMORY_PATH


class MemoryReader:
    """Reads persistent memory files for user/org context."""

    def read_user_memory(self) -> str:
        """Read USER_MEMORY.md contents."""
        return self._read_file(USER_MEMORY_PATH)

    def read_company_memory(self) -> str:
        """Read COMPANY_MEMORY.md contents."""
        return self._read_file(COMPANY_MEMORY_PATH)

    def _read_file(self, path) -> str:
        """Safely read a markdown file."""
        try:
            if path.exists():
                content = path.read_text(encoding="utf-8").strip()
                # Skip if file only has the header
                lines = [l for l in content.split("\n") if l.strip() and not l.startswith("#")]
                return "\n".join(lines)
            return ""
        except Exception as e:
            print(f"⚠️ Failed to read {path}: {e}")
            return ""