"""
Streamlit UI for Agentic RAG Chatbot.
Chat interface + PDF upload + citations display.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import tempfile
import os
from src.agents.orchestrator import DocumentAgent
from src.config import COMPLEXITY_LEVELS

# Page config
st.set_page_config(
    page_title="📚 Agentic RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

@st.cache_resource
def get_agent():
    return DocumentAgent()

agent = get_agent()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
if "doc_info" not in st.session_state:
    st.session_state.doc_info = None

# === Sidebar ===
with st.sidebar:
    st.header("📚 Agentic RAG Chatbot")
    st.caption("Upload a document and ask questions. Get grounded answers with citations.")

    st.divider()

    # File upload
    st.subheader("📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        help="Supported: PDF files (research papers, textbooks, reports)"
    )

    if uploaded_file:
        if st.button("📥 Ingest Document", use_container_width=True):
            with st.spinner("Parsing and indexing..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                try:
                    result = agent.ingest_document(tmp_path)
                    st.session_state.doc_id = result["doc_id"]
                    st.session_state.doc_info = result
                    st.success(
                        f"✅ Indexed **{result['title'][:60]}**\n\n"
                        f"📊 {result['chunks_indexed']} chunks | "
                        f"📑 {result['sections_found']} sections"
                    )
                except Exception as e:
                    st.error(f"❌ Ingestion failed: {e}")
                finally:
                    os.unlink(tmp_path)

    # Show current document info
    if st.session_state.doc_info:
        st.divider()
        st.subheader("📋 Current Document")
        info = st.session_state.doc_info
        st.write(f"**{info['title'][:60]}**")
        st.write(f"Chunks: {info['chunks_indexed']} | Sections: {info['sections_found']}")

    # Memory viewer
    st.divider()
    st.subheader("🧠 Memory")

    mem_tab1, mem_tab2 = st.tabs(["User", "Company"])
    with mem_tab1:
        user_mem = agent.mem_reader.read_user_memory()
        st.text(user_mem if user_mem else "No user memories yet.")
    with mem_tab2:
        company_mem = agent.mem_reader.read_company_memory()
        st.text(company_mem if company_mem else "No company memories yet.")

# === Main Chat Area ===
st.title("📚 Agentic RAG Chatbot")

if not st.session_state.doc_id:
    st.info("👈 Upload a PDF document in the sidebar to get started.")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show complexity summary if present
        if msg.get("complexity_summary"):
            cs = msg["complexity_summary"]
            if any(cs.values()):
                cols = st.columns(3)
                cols[0].metric("🔴 High", cs.get("high", 0))
                cols[1].metric("🟡 Medium", cs.get("medium", 0))
                cols[2].metric("🟢 Low", cs.get("low", 0))

        # Show citations if present
        if msg.get("citations"):
            with st.expander("📌 Citations & Sources"):
                for c in msg["citations"]:
                    cmplx_icon = COMPLEXITY_LEVELS.get(c.get("complexity", "low"), "⚪")
                    title = c.get("section_title", "")[:50]
                    st.write(
                        f"{cmplx_icon} Section {c['section']}, "
                        f"Page {c['page']} — *{c['section_type']}*"
                        f"{f' — {title}' if title else ''}"
                    )

# Chat input
if prompt := st.chat_input("Ask about your document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                result = agent.process_query(
                    query=prompt,
                    doc_id=st.session_state.doc_id
                )

                # Display answer
                st.markdown(result["answer"])

                # Complexity summary
                cs = result["complexity_summary"]
                if any(cs.values()):
                    cols = st.columns(3)
                    cols[0].metric("🔴 High Complexity", cs.get("high", 0))
                    cols[1].metric("🟡 Medium", cs.get("medium", 0))
                    cols[2].metric("🟢 Low", cs.get("low", 0))

                # Citations
                if result["citations"]:
                    with st.expander("📌 Citations & Sources"):
                        for c in result["citations"]:
                            cmplx_icon = COMPLEXITY_LEVELS.get(c.get("complexity", "low"), "⚪")
                            title = c.get("section_title", "")[:50]
                            st.write(
                                f"{cmplx_icon} Section {c['section']}, "
                                f"Page {c['page']} — *{c['section_type']}*"
                                f"{f' — {title}' if title else ''}"
                            )

                # Agent trace (debug)
                with st.expander("🔍 Agent Trace"):
                    for step in result["agent_trace"]["steps"]:
                        st.json(step)

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "complexity_summary": result["complexity_summary"],
                    "citations": result["citations"]
                })

            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Sorry, I encountered an error: {e}"
                })