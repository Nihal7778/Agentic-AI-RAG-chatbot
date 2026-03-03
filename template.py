import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

list_of_files = [
    "",
    "src/config.py",
    "src/__init__.py",
    "src/ingestion/__init__.py",
    "src/ingestion/parser.py",
    "src/ingestion/chunker.py",
    "src/retrieval/__init__.py",
    "src/retrieval/embedder.py",
    "src/retrieval/basic_rag.py",
    "src/retrieval/hyde_rag.py",
    "src/agents/__init__.py",
    "src/agents/router.py",
    "src/agents/evaluator.py",
    "src/agents/risk_scorer.py",
    "src/agents/orchestrator.py",
    "src/generation/__init__.py",
    "src/generation/generator.py",
    "src/memory/__init__.py",
    "src/memory/reader.py",
    "src/memory/writer.py",
    "src/security/__init__.py",
    "src/security/pii_filter.py",
    "ui/streamlit_app.py",
    "ui/style.css",
    "ui/chat.html",
    "src/main.py",
    "tests/__init__.py",
    "tests/test_sanity.py",
    ".env",
    "setup.py",
    "app.py",
    "research/trials.ipynb",
    "sample_docs/Data",
    "requirements.txt",
    "src/tools/weather.py",
    "src/tools/__init__.py",
    "tests/eval_harness.py",
]

for file_path in list_of_files:
    filepath = Path(file_path)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Created directory; {filedir} for file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Created empty file: {filepath}")