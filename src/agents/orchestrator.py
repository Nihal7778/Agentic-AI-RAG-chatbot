"""
Orchestrator for Agentic RAG Chatbot.
Main agent loop: route → retrieve → evaluate → score → generate → memory.
Includes conversation history, multimodal support, and document management.
"""

from typing import Dict, Optional, List
from src.retrieval.embedder import ContractEmbedder
from src.retrieval.basic_rag import BasicRAGRetriever
from src.retrieval.hyde_rag import MultiHyDERetriever
from src.agents.router import QueryRouter
from src.agents.evaluator import RetrievalEvaluator
from src.agents.risk_scorer import ComplexityScorer
from src.generation.generator import ResponseGenerator
from src.memory.reader import MemoryReader
from src.memory.writer import MemoryWriter
from src.tools.weather import WeatherTool
from src.config import MAX_RETRIES, TOP_K, MAX_CONVERSATION_HISTORY, MULTIMODAL_ENABLED

if MULTIMODAL_ENABLED:
    from src.retrieval.clip_embedder import CLIPEmbedder
    from src.retrieval.multimodal_rag import MultimodalRetriever
    from src.ingestion.image_extractor import (
        extract_images, extract_tables,
        prepare_images_for_indexing, prepare_tables_for_indexing,
    )


class DocumentAgent:
    """
    Main agent that orchestrates the full pipeline.

    Flow:
    1. Check conversation history for context
    2. Read user memory
    3. Route query → conversational / simple / complex / image_query / tool_call
    4. Retrieve documents (Basic RAG, Multi-HyDE, or Multimodal)
    5. Evaluate results → retry if insufficient (agentic loop)
    6. Score complexity of retrieved sections
    7. Generate response with citations
    8. Write to memory if high-signal
    """

    def __init__(self):
        self.embedder = ContractEmbedder()
        self.basic_rag = BasicRAGRetriever(self.embedder)
        self.hyde_rag = MultiHyDERetriever(self.embedder)
        self.router = QueryRouter()
        self.evaluator = RetrievalEvaluator()
        self.complexity_scorer = ComplexityScorer()
        self.generator = ResponseGenerator()
        self.mem_reader = MemoryReader()
        self.mem_writer = MemoryWriter()
        self.weather_tool = WeatherTool()
        self.conversation_history: List[Dict] = []

        # Multimodal components
        self.multimodal_enabled = MULTIMODAL_ENABLED
        self.clip_embedder = None
        self.multimodal_retriever = None
        self.extracted_images_cache: Dict[str, list] = {}

        if self.multimodal_enabled:
            try:
                self.clip_embedder = CLIPEmbedder()
                self.multimodal_retriever = MultimodalRetriever(
                    self.embedder, self.clip_embedder
                )
            except Exception as e:
                print(f"⚠️ Multimodal init failed, text-only mode: {e}")
                self.multimodal_enabled = False

    def process_query(
        self,
        query: str,
        doc_id: Optional[str] = None
    ) -> Dict:
        trace = {"steps": []}

        # --- Step 1: Build conversation context ---
        conv_context = self._get_conversation_context()
        trace["steps"].append({
            "step": "conversation_history",
            "turns": len(self.conversation_history)
        })

        # --- Step 2: Read memory ---
        user_context = self.mem_reader.read_user_memory()
        company_context = self.mem_reader.read_company_memory()
        trace["steps"].append({
            "step": "memory_read",
            "user_context": user_context[:200] if user_context else "none",
        })

        # --- Step 3: Route the query ---
        route = self.router.classify(query)
        trace["steps"].append({
            "step": "route",
            "strategy": route["strategy"],
            "is_technical_query": route["is_technical_query"],
            "method": route["method"]
        })

        # --- Conversational shortcut ---
        if route["strategy"] == "conversational":
            chat_response = self._handle_conversation(query)
            self._add_to_history(query, chat_response)
            trace["steps"].append({"step": "conversational_response"})
            return {
                "answer": chat_response,
                "citations": [],
                "complexity_summary": {"high": 0, "medium": 0, "low": 0},
                "agent_trace": trace
            }

        # --- Tool call shortcut (Feature C: Open-Meteo) ---
        if route["strategy"] == "tool_call":
            location = route.get("location", "")
            trace["steps"].append({"step": "tool_call", "tool": "weather", "location": location})
            tool_result = self.weather_tool.analyze(location)
            self._add_to_history(query, tool_result["answer"])
            return {
                "answer": tool_result["answer"],
                "citations": [],
                "complexity_summary": {"high": 0, "medium": 0, "low": 0},
                "agent_trace": trace
            }

        # --- Check if documents exist ---
        if self.embedder.get_collection_count() == 0:
            self._add_to_history(query, "Please upload a document first.")
            return {
                "answer": "I don't have any documents loaded yet. Upload a PDF in the sidebar and I'll help you explore it! 📄",
                "citations": [],
                "complexity_summary": {"high": 0, "medium": 0, "low": 0},
                "agent_trace": trace
            }

        filter_dict = {"doc_id": doc_id} if doc_id else None

        # --- Image query path (multimodal) ---
        if route["strategy"] == "image_query" and self.multimodal_enabled:
            return self._handle_image_query(
                query, filter_dict, user_context,
                company_context, conv_context, trace
            )

        # --- Step 4: Retrieve with retry loop (text path) ---
        documents = []
        enhanced_query = self._enhance_with_context(query, conv_context)

        for attempt in range(MAX_RETRIES + 1):
            current_query = enhanced_query

            if route["strategy"] == "simple":
                result = self.basic_rag.retrieve(
                    current_query, k=TOP_K, filter_dict=filter_dict
                )
            else:
                result = self.hyde_rag.retrieve(
                    current_query, k=TOP_K,
                    filter_dict=filter_dict,
                    is_technical_query=route["is_technical_query"]
                )

            documents = result["documents"]

            evaluation = self.evaluator.evaluate(current_query, documents)
            trace["steps"].append({
                "step": f"retrieve_attempt_{attempt + 1}",
                "strategy": result["strategy"],
                "num_docs": len(documents),
                "avg_score": evaluation["avg_score"],
                "sufficient": evaluation["sufficient"]
            })

            if evaluation["sufficient"]:
                break

            if attempt < MAX_RETRIES:
                enhanced_query = self.evaluator.refine_query(
                    enhanced_query, evaluation.get("reason", "")
                )
                trace["steps"].append({
                    "step": "query_refined",
                    "new_query": enhanced_query
                })

        # --- Step 5: Score complexity ---
        scored_docs = self.complexity_scorer.score_sections(
            documents, user_query=query
        )
        trace["steps"].append({
            "step": "complexity_scoring",
            "high": sum(1 for d in scored_docs if d.get("complexity") == "high"),
            "medium": sum(1 for d in scored_docs if d.get("complexity") == "medium"),
            "low": sum(1 for d in scored_docs if d.get("complexity") == "low"),
        })

        # --- Step 6: Generate response ---
        response = self.generator.generate(
            query=query,
            documents=scored_docs,
            user_context=user_context,
            company_context=company_context,
            conversation_history=conv_context,
            strategy=route["strategy"]
        )
        trace["steps"].append({
            "step": "generate",
            "has_citations": len(response.get("citations", [])) > 0
        })

        # --- Step 7: Write to memory ---
        mem_result = self.mem_writer.decide_and_write(
            query=query,
            response=response["answer"],
            documents=scored_docs,
            user_context=user_context
        )
        trace["steps"].append({
            "step": "memory_write",
            "wrote_user": mem_result.get("wrote_user", False),
            "wrote_company": mem_result.get("wrote_company", False)
        })

        self._add_to_history(query, response["answer"])

        return {
            "answer": response["answer"],
            "citations": response["citations"],
            "complexity_summary": response["complexity_summary"],
            "agent_trace": trace
        }

    # === Multimodal Handling ===

    def _handle_image_query(
        self, query, filter_dict, user_context,
        company_context, conv_context, trace
    ) -> Dict:
        """Handle image/figure/chart queries via dual pipeline."""

        # Gather all cached PIL images across documents
        cached_images = []
        for imgs in self.extracted_images_cache.values():
            cached_images.extend(imgs)

        mm_result = self.multimodal_retriever.retrieve_with_descriptions(
            query=query,
            extracted_images=cached_images,
            k_text=TOP_K,
            k_images=2,
            filter_dict=filter_dict,
        )

        trace["steps"].append({
            "step": "multimodal_retrieval",
            "text_results": len(mm_result["text"]),
            "image_results": len(mm_result["images"]),
            "has_images": mm_result["has_images"],
        })

        # Build extra context from image descriptions
        extra_context = ""
        if mm_result["image_descriptions"]:
            extra_context = (
                "\n\n--- Image Descriptions ---\n\n"
                + "\n\n".join(mm_result["image_descriptions"])
            )

        # Score text docs for complexity
        scored_docs = self.complexity_scorer.score_sections(
            mm_result["text"], user_query=query
        )

        # Generate response with text + image context
        response = self.generator.generate(
            query=query,
            documents=scored_docs,
            user_context=user_context,
            company_context=company_context,
            conversation_history=conv_context,
            strategy="image_query",
            extra_context=extra_context,
        )

        trace["steps"].append({
            "step": "generate",
            "has_citations": len(response.get("citations", [])) > 0,
            "used_vision": mm_result["has_images"],
        })

        # Memory write
        mem_result = self.mem_writer.decide_and_write(
            query=query,
            response=response["answer"],
            documents=scored_docs,
            user_context=user_context
        )
        trace["steps"].append({
            "step": "memory_write",
            "wrote_user": mem_result.get("wrote_user", False),
            "wrote_company": mem_result.get("wrote_company", False)
        })

        self._add_to_history(query, response["answer"])

        return {
            "answer": response["answer"],
            "citations": response["citations"],
            "complexity_summary": response.get("complexity_summary", {}),
            "has_images": mm_result["has_images"],
            "agent_trace": trace,
        }

    # === Document Management ===

    def ingest_document(self, pdf_path: str) -> Dict:
        """Ingest a PDF — text + images + tables."""
        from src.ingestion.parser import parse_document
        from src.ingestion.chunker import chunk_document, chunks_to_documents

        # Text pipeline (existing)
        document = parse_document(pdf_path)
        chunks = chunk_document(document)
        doc_id = chunks[0].doc_id if chunks else "unknown"
        docs = chunks_to_documents(chunks)
        text_count = self.embedder.index_chunks(docs)

        image_count = 0
        table_count = 0

        # Multimodal pipeline (new, additive)
        if self.multimodal_enabled and self.clip_embedder:
            try:
                # Images → CLIP → image_chunks collection
                raw_images = extract_images(pdf_path, doc_id)
                if raw_images:
                    prepared = prepare_images_for_indexing(raw_images, self.clip_embedder)
                    image_count = self.clip_embedder.index_images(prepared)
                    self.extracted_images_cache[doc_id] = raw_images

                # Tables → markdown → text collection (MiniLM)
                raw_tables = extract_tables(pdf_path, doc_id)
                if raw_tables:
                    table_docs = prepare_tables_for_indexing(raw_tables)
                    table_count = self.embedder.index_chunks(table_docs)

            except Exception as e:
                print(f"⚠️ Multimodal ingestion failed (text still indexed): {e}")

        return {
            "doc_id": doc_id,
            "chunks_indexed": text_count,
            "images_indexed": image_count,
            "tables_indexed": table_count,
            "sections_found": len(document.sections),
            "title": document.title,
        }

    def delete_document(self, doc_id: str) -> Dict:
        """Delete all chunks belonging to a document."""
        self.embedder.delete_document(doc_id)
        if self.multimodal_enabled and self.clip_embedder:
            try:
                self.clip_embedder.collection.delete(where={"doc_id": doc_id})
            except Exception:
                pass
        self.extracted_images_cache.pop(doc_id, None)
        return {"deleted": True, "doc_id": doc_id}

    def list_documents(self) -> Dict:
        """List all indexed documents."""
        count = self.embedder.get_collection_count()
        image_count = 0
        if self.multimodal_enabled and self.clip_embedder:
            image_count = self.clip_embedder.get_collection_count()
        return {"total_chunks": count, "total_images": image_count}

    def reset_all(self) -> Dict:
        """Clear all documents, memory, and conversation history."""
        self.embedder.reset()
        if self.multimodal_enabled and self.clip_embedder:
            self.clip_embedder.reset()
        self.conversation_history = []
        self.extracted_images_cache = {}
        return {"reset": True}

    # === Conversation History ===

    def _add_to_history(self, query: str, answer: str):
        self.conversation_history.append({
            "query": query,
            "answer": answer[:300]
        })
        if len(self.conversation_history) > MAX_CONVERSATION_HISTORY:
            self.conversation_history = self.conversation_history[-MAX_CONVERSATION_HISTORY:]

    def _get_conversation_context(self) -> str:
        if not self.conversation_history:
            return ""
        lines = []
        for turn in self.conversation_history[-5:]:
            lines.append(f"User: {turn['query']}")
            lines.append(f"Assistant: {turn['answer'][:150]}")
        return "\n".join(lines)

    def _enhance_with_context(self, query: str, conv_context: str) -> str:
        ambiguous = ["it", "this", "that", "these", "those", "them",
                     "the same", "more about", "what about", "elaborate"]
        q_lower = query.lower()
        if conv_context and any(word in q_lower for word in ambiguous):
            last_turn = self.conversation_history[-1] if self.conversation_history else None
            if last_turn:
                return f"{query} (context: previously discussed {last_turn['query']})"
        return query

    def _handle_conversation(self, query: str) -> str:
        q = query.lower().strip()

        greetings = ["hey", "hello", "hi", "howdy", "sup", "yo"]
        how_are_you = ["how are you", "how's it going", "what's up", "how do you do"]
        thanks = ["thanks", "thank you", "thx"]
        byes = ["bye", "goodbye", "see you"]

        if any(g in q for g in greetings):
            has_docs = self.embedder.get_collection_count() > 0
            if has_docs:
                return "Hey! 👋 I have a document loaded and ready. Ask me anything about it!"
            return "Hey! 👋 Upload a PDF document and I'll help you explore it. Ask me anything about its content!"

        if any(h in q for h in how_are_you):
            return "I'm doing great, thanks for asking! 😊 How can I help you today?"

        if any(t in q for t in thanks):
            return "You're welcome! Let me know if you have more questions. 😊"

        if any(b in q for b in byes):
            return "Goodbye! Feel free to come back anytime. 👋"

        if "what can you do" in q or "help" in q:
            return ("I'm an AI document assistant! Here's what I can do:\n"
                    "📄 Analyze uploaded PDF documents\n"
                    "🔍 Answer questions grounded in the document content\n"
                    "🖼️ Understand figures, charts, and diagrams\n"
                    "📌 Provide citations for every answer\n"
                    "🧠 Remember your preferences across conversations\n\n"
                    "Upload a PDF and start asking questions!")

        if "who are you" in q:
            return "I'm an Agentic RAG Chatbot — I answer questions based on documents you upload. I use HyDE retrieval for complex queries, CLIP for image understanding, and provide cited, grounded answers."

        return "I'm here to help with document questions! Upload a PDF or ask me something about a loaded document. 😊"