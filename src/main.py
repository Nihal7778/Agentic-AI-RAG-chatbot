"""
FastAPI backend + frontend for Agentic RAG Chatbot.
Single entry point — serves both API and HTML UI.
Includes streaming, document management, and conversation history.

Run: uvicorn src.main:app --reload --port 8000
Open: http://localhost:8000
"""

import os
import json
import tempfile
import asyncio
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional
from src.agents.orchestrator import DocumentAgent

app = FastAPI(title="Agentic RAG Chatbot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
UI_DIR = Path(__file__).parent.parent / "ui"
app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")

# Initialize agent
agent = DocumentAgent()


class QueryRequest(BaseModel):
    query: str
    doc_id: Optional[str] = None
    stream: Optional[bool] = False


class QueryResponse(BaseModel):
    answer: str
    citations: list
    complexity_summary: dict
    agent_trace: dict


# === Frontend ===
@app.get("/")
def serve_frontend():
    return FileResponse(str(UI_DIR / "chat.html"))


# === Health ===
@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks_indexed": agent.embedder.get_collection_count(),
        "conversation_turns": len(agent.conversation_history)
    }


# === Upload & Ingest ===
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and ingest a PDF document."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files supported")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = agent.ingest_document(tmp_path)
        return result
    except Exception as e:
        raise HTTPException(500, f"Ingestion failed: {e}")
    finally:
        os.unlink(tmp_path)


# === Query (normal) ===
@app.post("/query", response_model=QueryResponse)
def query_document(req: QueryRequest):
    """Ask a question about an uploaded document."""
    try:
        result = agent.process_query(query=req.query, doc_id=req.doc_id)
        return result
    except Exception as e:
        raise HTTPException(500, f"Query failed: {e}")


# === Query (streaming) ===
@app.post("/query/stream")
async def query_document_stream(req: QueryRequest):
    """Stream the answer token by token."""
    try:
        result = agent.process_query(query=req.query, doc_id=req.doc_id)

        async def stream_response():
            # First stream the answer word by word
            words = result["answer"].split(" ")
            for i, word in enumerate(words):
                chunk = {"type": "token", "content": word + " "}
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.03)  # 30ms per word

            # Then send metadata
            meta = {
                "type": "done",
                "citations": result["citations"],
                "complexity_summary": result["complexity_summary"],
                "agent_trace": result["agent_trace"]
            }
            yield f"data: {json.dumps(meta)}\n\n"

        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(500, f"Query failed: {e}")


# === Memory ===
@app.get("/memory")
def get_memory():
    """Read current memory files."""
    return {
        "user_memory": agent.mem_reader.read_user_memory(),
        "company_memory": agent.mem_reader.read_company_memory()
    }


# === Document Management ===
@app.get("/documents")
def list_documents():
    """List indexed document stats."""
    return agent.list_documents()


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    """Delete all chunks for a specific document."""
    try:
        return agent.delete_document(doc_id)
    except Exception as e:
        raise HTTPException(500, f"Delete failed: {e}")


@app.post("/documents/reset")
def reset_all():
    """Clear all documents, memory, and conversation history."""
    try:
        return agent.reset_all()
    except Exception as e:
        raise HTTPException(500, f"Reset failed: {e}")


# === Conversation History ===
@app.get("/history")
def get_history():
    """Get conversation history."""
    return {"history": agent.conversation_history}


@app.delete("/history")
def clear_history():
    """Clear conversation history."""
    agent.conversation_history = []
    return {"cleared": True}