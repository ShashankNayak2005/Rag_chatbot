"""
app.py
Professional Document Question-Answering Chatbot using RAG.
Features:
- Minimal, modern AI-chat interface using native Streamlit chat components.
- Local Sentence Transformers embeddings (all-MiniLM-L6-v2) and ChromaDB persistent vector storage.
- Strict anti-hallucination evidence validation (ANSWER_NOT_FOUND_IN_DOCUMENT).
- Natural language generation via LLM API and zero-cost local extractive fallback.
- Compact citation badges, expandable retrieval diagnostics, and chat history export.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import inspect
import importlib
import html
import streamlit as st

from src.document_loader import (
    load_document,
    DocumentValidationError,
    EmptyFileError,
    FileTooLargeError,
    UnsupportedFileTypeError,
    UnreadableFileError,
    NoExtractableTextError,
)
from src.text_chunker import chunk_document
from src.embedding_service import EmbeddingService
from src.vector_store import ChromaVectorStore
from src.retriever import SemanticRetriever

import src.config
importlib.reload(src.config)
src.config.get_settings.cache_clear()
settings = src.config.get_settings()

import src.qa_engine
importlib.reload(src.qa_engine)
from src.qa_engine import QAEngine, ANSWER_NOT_FOUND_IN_DOCUMENT


# Path for persistent conversation history
CONVERSATION_HISTORY_FILE = Path("data/conversation_history.json")


def load_persisted_conversation() -> List[Dict[str, Any]]:
    """Load conversation history from disk if available."""
    if CONVERSATION_HISTORY_FILE.exists():
        try:
            return json.loads(CONVERSATION_HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save_persisted_conversation(messages: List[Dict[str, Any]]) -> None:
    """Save conversation history to disk."""
    try:
        CONVERSATION_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        serializable = []
        for m in messages:
            serializable.append({
                "role": m.get("role"),
                "content": m.get("content"),
                "timestamp": m.get("timestamp"),
                "response_data": m.get("response_data"),
            })
        CONVERSATION_HISTORY_FILE.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    except Exception:
        pass


def export_conversation_markdown(messages: List[Dict[str, Any]], doc_info: Optional[Dict[str, Any]]) -> str:
    """Generate a clean Markdown transcript of the conversation with citations."""
    doc_name = doc_info.get("filename", "No Document") if doc_info else "No Document"
    doc_id = doc_info.get("document_id", "N/A") if doc_info else "N/A"
    lines = [
        "# 📄 Document Q&A — Conversation Transcript",
        f"- **Active Document:** `{doc_name}`",
        f"- **Document ID:** `{doc_id}`",
        f"- **Exported At:** {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}",
        f"- **Total Messages:** {len(messages)}",
        "\n---\n",
    ]
    turn_num = 1
    for i in range(0, len(messages), 2):
        u_msg = messages[i]
        a_msg = messages[i + 1] if i + 1 < len(messages) else None
        lines.append(f"### Turn {turn_num}")
        lines.append(f"**🧑‍💻 User ({u_msg.get('timestamp', '')}):**\n> {u_msg.get('content', '')}\n")
        if a_msg:
            resp = a_msg.get("response_data", {})
            lines.append(f"**🤖 Assistant ({a_msg.get('timestamp', '')}):**\n")
            lines.append(f"{a_msg.get('content', '')}\n")
            if resp.get("is_found"):
                lines.append(
                    f"- **Source Citation:** `{resp.get('source_filename', 'N/A')}` | "
                    f"Page {resp.get('page_number', 'N/A')} | "
                    f"Chunk #{resp.get('chunk_id', 'N/A')} | "
                    f"Cosine Distance: `{resp.get('distance', 0.0):.4f}`"
                )
                if resp.get("supporting_passage"):
                    lines.append(f"- **Evidence Passage:** *\"{resp.get('supporting_passage').strip()}\"*")
            elif resp.get("status") == "not_found":
                lines.append(f"- **Grounding Status:** `{ANSWER_NOT_FOUND_IN_DOCUMENT}` (Guaranteed unhallucinated)")
            elif resp.get("status") == "api_error":
                lines.append(f"- **API Error:** {resp.get('error_message')}")
        lines.append("\n---\n")
        turn_num += 1
    return "\n".join(lines)


# Configure page settings
st.set_page_config(
    page_title="Document Q&A Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Cache expensive resources globally across all sessions and reruns
@st.cache_resource(show_spinner=False)
def get_shared_embedding_service(model_name: str) -> EmbeddingService:
    """Singleton cached embedding service."""
    return EmbeddingService(model_name)


@st.cache_resource(show_spinner=False)
def get_shared_vector_store(persist_dir: str) -> ChromaVectorStore:
    """Singleton cached ChromaDB store."""
    return ChromaVectorStore(persist_directory=persist_dir)


# Initialize session states
if "qa_engine" not in st.session_state:
    st.session_state.qa_engine = None
if "current_doc_info" not in st.session_state:
    st.session_state.current_doc_info = None
if "sample_selected" not in st.session_state:
    st.session_state.sample_selected = None
if "sample_filename" not in st.session_state:
    st.session_state.sample_filename = None
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "messages" not in st.session_state:
    st.session_state.messages = load_persisted_conversation()
if "pending_chat_prompt" not in st.session_state:
    st.session_state.pending_chat_prompt = None
if "embedding_service" not in st.session_state:
    try:
        st.session_state.embedding_service = get_shared_embedding_service(settings.embedding_model)
    except Exception:
        st.session_state.embedding_service = None
if "vector_store" not in st.session_state:
    try:
        st.session_state.vector_store = get_shared_vector_store(settings.chroma_persist_directory)
    except Exception:
        st.session_state.vector_store = None

# Upgrade cached qa_engine instance if created with older class definition
if st.session_state.qa_engine is not None:
    try:
        sig = inspect.signature(st.session_state.qa_engine.answer_question)
        if "conversation_history" not in sig.parameters:
            old_engine = st.session_state.qa_engine
            st.session_state.qa_engine = QAEngine(
                retriever=getattr(old_engine, "retriever", None),
                api_key=getattr(old_engine, "api_key", None),
                base_url=getattr(old_engine, "base_url", None),
                model=getattr(old_engine, "model", None),
                distance_threshold=getattr(old_engine, "distance_threshold", None),
            )
    except Exception:
        pass

# Clean, professional styling: subtle borders, light surfaces, refined chat spacing
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Chat workspace container: generous top clearance so header is never obscured by Streamlit toolbar */
    .block-container {
        padding-top: 5rem !important;
        padding-bottom: 2.5rem;
        max-width: 980px;
    }

    /* Style Streamlit top header toolbar safely */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* Native chat message bubble enhancements */
    div[data-testid="stChatMessage"] {
        padding: 1rem 1.2rem;
        margin-bottom: 0.85rem;
        border-radius: 12px;
        line-height: 1.6;
        transition: all 0.15s ease-in-out;
    }

    /* User messages: light gray background, subtle border, right alignment feel */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        margin-left: 10%;
    }

    /* Assistant messages: clean crisp white, subtle border */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        margin-right: 5%;
    }

    /* Compact citation metadata tags */
    .cite-pill-row {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 6px;
        margin-top: 10px;
        margin-bottom: 2px;
    }
    .cite-pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        background: #f1f5f9;
        color: #334155;
        border: 1px solid #e2e8f0;
    }
    .cite-pill-sim {
        background: #ecfdf5;
        color: #047857;
        border-color: #a7f3d0;
    }

    /* Unified Header Box */
    .header-box {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .header-left {
        flex: 1 1 320px;
    }
    .header-title-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 3px;
    }
    .header-title-text {
        margin: 0;
        font-weight: 800;
        font-size: 1.45rem;
        color: #0f172a;
        letter-spacing: -0.3px;
        line-height: 1.2;
    }
    .header-badge-tag {
        font-size: 0.72rem;
        font-weight: 700;
        background: #eff6ff;
        color: #2563eb;
        border: 1px solid #bfdbfe;
        padding: 2px 8px;
        border-radius: 9999px;
        letter-spacing: 0.2px;
    }
    .header-desc {
        margin: 0;
        color: #64748b;
        font-size: 0.84rem;
        line-height: 1.45;
    }
    .header-right {
        flex: 0 0 auto;
    }
    .header-status-card {
        border-radius: 8px;
        padding: 8px 14px;
        text-align: right;
        min-width: 175px;
    }
    .header-status-card.active {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
    }
    .header-status-card.active .status-label {
        font-size: 0.68rem;
        font-weight: 700;
        color: #166534;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .header-status-card.active .status-filename {
        font-size: 0.88rem;
        font-weight: 700;
        color: #15803d;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 220px;
        margin: 2px 0;
    }
    .header-status-card.active .status-chunks {
        font-size: 0.72rem;
        color: #16a34a;
    }
    .header-status-card.empty {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
    }
    .header-status-card.empty .status-label {
        font-size: 0.68rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .header-status-card.empty .status-filename {
        font-size: 0.82rem;
        font-weight: 600;
        color: #475569;
        margin: 2px 0;
    }
    .header-status-card.empty .status-chunks {
        font-size: 0.70rem;
        color: #94a3b8;
    }

    /* Attractive, Highly Visible Sidebar Access Button */
    div[data-testid="collapsedControl"] {
        display: flex !important;
        align-items: center !important;
        top: 0.85rem !important;
        left: 1rem !important;
        z-index: 999999 !important;
        visibility: visible !important;
    }

    div[data-testid="collapsedControl"] button,
    button[data-testid="stSidebarCollapsedControl"] {
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 6px !important;
        background: #2563eb !important;
        color: #ffffff !important;
        border: 1px solid #1d4ed8 !important;
        border-radius: 9px !important;
        padding: 7px 14px !important;
        font-weight: 700 !important;
        font-size: 0.82rem !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
        cursor: pointer !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    div[data-testid="collapsedControl"] button svg,
    button[data-testid="stSidebarCollapsedControl"] svg {
        width: 18px !important;
        height: 18px !important;
        stroke: #ffffff !important;
        fill: #ffffff !important;
    }

    div[data-testid="collapsedControl"] button::after,
    button[data-testid="stSidebarCollapsedControl"]::after {
        content: "📂 Document Workspace" !important;
        font-size: 0.80rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        letter-spacing: 0.2px !important;
        white-space: nowrap !important;
    }

    div[data-testid="collapsedControl"] button:hover,
    button[data-testid="stSidebarCollapsedControl"]:hover {
        background: #1d4ed8 !important;
        transform: translateY(-1px) scale(1.02) !important;
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.45) !important;
    }

    button[data-testid="stSidebarCollapseButton"] {
        border-radius: 8px !important;
        padding: 6px !important;
        transition: all 0.15s ease !important;
    }
    button[data-testid="stSidebarCollapseButton"]:hover {
        background: rgba(37, 99, 235, 0.08) !important;
        color: #2563eb !important;
    }

    /* Dark Theme Support for Header Box */
    @media (prefers-color-scheme: dark) {
        .header-box {
            background: #1e293b !important;
            border-color: #334155 !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25) !important;
        }
        .header-title-text { color: #f8fafc !important; }
        .header-badge-tag {
            background: rgba(37, 99, 235, 0.2) !important;
            color: #60a5fa !important;
            border-color: rgba(37, 99, 235, 0.4) !important;
        }
        .header-desc { color: #94a3b8 !important; }
        .header-status-card.active {
            background: rgba(16, 185, 129, 0.15) !important;
            border-color: rgba(16, 185, 129, 0.35) !important;
        }
        .header-status-card.active .status-label { color: #34d399 !important; }
        .header-status-card.active .status-filename { color: #f0fdf4 !important; }
        .header-status-card.active .status-chunks { color: #a7f3d0 !important; }
        .header-status-card.empty {
            background: #0f172a !important;
            border-color: #334155 !important;
        }
        .header-status-card.empty .status-label { color: #94a3b8 !important; }
        .header-status-card.empty .status-filename { color: #cbd5e1 !important; }
        .header-status-card.empty .status-chunks { color: #64748b !important; }
    }

    [data-theme="dark"] .header-box, .stApp[data-theme="dark"] .header-box {
        background: #1e293b !important;
        border-color: #334155 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25) !important;
    }
    [data-theme="dark"] .header-title-text, .stApp[data-theme="dark"] .header-title-text { color: #f8fafc !important; }
    [data-theme="dark"] .header-badge-tag, .stApp[data-theme="dark"] .header-badge-tag {
        background: rgba(37, 99, 235, 0.2) !important;
        color: #60a5fa !important;
        border-color: rgba(37, 99, 235, 0.4) !important;
    }
    [data-theme="dark"] .header-desc, .stApp[data-theme="dark"] .header-desc { color: #94a3b8 !important; }
    [data-theme="dark"] .header-status-card.active, .stApp[data-theme="dark"] .header-status-card.active {
        background: rgba(16, 185, 129, 0.15) !important;
        border-color: rgba(16, 185, 129, 0.35) !important;
    }
    [data-theme="dark"] .header-status-card.active .status-label, .stApp[data-theme="dark"] .header-status-card.active .status-label { color: #34d399 !important; }
    [data-theme="dark"] .header-status-card.active .status-filename, .stApp[data-theme="dark"] .header-status-card.active .status-filename { color: #f0fdf4 !important; }
    [data-theme="dark"] .header-status-card.active .status-chunks, .stApp[data-theme="dark"] .header-status-card.active .status-chunks { color: #a7f3d0 !important; }
    [data-theme="dark"] .header-status-card.empty, .stApp[data-theme="dark"] .header-status-card.empty {
        background: #0f172a !important;
        border-color: #334155 !important;
    }
    [data-theme="dark"] .header-status-card.empty .status-label, .stApp[data-theme="dark"] .header-status-card.empty .status-label { color: #94a3b8 !important; }
    [data-theme="dark"] .header-status-card.empty .status-filename, .stApp[data-theme="dark"] .header-status-card.empty .status-filename { color: #cbd5e1 !important; }
    [data-theme="dark"] .header-status-card.empty .status-chunks, .stApp[data-theme="dark"] .header-status-card.empty .status-chunks { color: #64748b !important; }

    /* =========================================================================
       HIGH-VISIBILITY CHAT ENTER BOX (BOLD HIGHLIGHTED BORDER & GLOW)
       ========================================================================= */
    div[data-testid="stBottom"], footer {
        background: transparent !important;
    }

    div[data-testid="stChatInput"],
    .stChatInput {
        border-radius: 16px !important;
        border: 2.5px solid #2563eb !important;
        background-color: #ffffff !important;
        box-shadow: 0 4px 18px rgba(37, 99, 235, 0.20), 0 0 0 1px rgba(37, 99, 235, 0.25) !important;
        padding: 4px 8px !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }

    div[data-testid="stChatInput"] > div {
        border: none !important;
        background: transparent !important;
    }

    div[data-testid="stChatInput"]:hover,
    .stChatInput:hover {
        border-color: #1d4ed8 !important;
        box-shadow: 0 6px 24px rgba(37, 99, 235, 0.28), 0 0 0 2px rgba(37, 99, 235, 0.35) !important;
    }

    div[data-testid="stChatInput"]:focus-within,
    .stChatInput:focus-within {
        border-color: #1d4ed8 !important;
        box-shadow: 0 8px 30px rgba(37, 99, 235, 0.35), 0 0 0 3.5px rgba(37, 99, 235, 0.32) !important;
        transform: translateY(-2px) !important;
    }

    div[data-testid="stChatInput"] textarea,
    .stChatInput textarea {
        font-size: 0.96rem !important;
        font-weight: 500 !important;
        color: #0f172a !important;
        background: transparent !important;
        line-height: 1.5 !important;
    }

    div[data-testid="stChatInput"] textarea::placeholder,
    .stChatInput textarea::placeholder {
        color: #64748b !important;
        font-weight: 500 !important;
        font-size: 0.92rem !important;
    }

    div[data-testid="stChatInput"] button[data-testid="stChatInputSubmitButton"],
    .stChatInput button[data-testid="stChatInputSubmitButton"] {
        background: #2563eb !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        transition: all 0.2s ease !important;
        padding: 6px !important;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.35) !important;
    }

    div[data-testid="stChatInput"] button[data-testid="stChatInputSubmitButton"]:hover:not(:disabled),
    .stChatInput button[data-testid="stChatInputSubmitButton"]:hover:not(:disabled) {
        background: #1d4ed8 !important;
        transform: scale(1.08) !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.5) !important;
    }

    div[data-testid="stChatInput"] button[data-testid="stChatInputSubmitButton"] svg,
    .stChatInput button[data-testid="stChatInputSubmitButton"] svg {
        fill: #ffffff !important;
        stroke: #ffffff !important;
    }

    div[data-testid="stChatInput"]:has(textarea:disabled),
    .stChatInput:has(textarea:disabled) {
        border: 2.5px solid #2563eb !important;
        background-color: #ffffff !important;
        box-shadow: 0 4px 18px rgba(37, 99, 235, 0.20), 0 0 0 1px rgba(37, 99, 235, 0.25) !important;
        opacity: 1 !important;
    }

    /* Dark Mode High-Visibility Support */
    @media (prefers-color-scheme: dark) {
        div[data-testid="stChatInput"],
        .stChatInput {
            border-color: #3b82f6 !important;
            background-color: #1e293b !important;
            box-shadow: 0 4px 22px rgba(59, 130, 246, 0.32), 0 0 0 1px rgba(59, 130, 246, 0.35) !important;
        }
        div[data-testid="stChatInput"]:focus-within,
        .stChatInput:focus-within {
            border-color: #60a5fa !important;
            box-shadow: 0 8px 32px rgba(59, 130, 246, 0.42), 0 0 0 3.5px rgba(59, 130, 246, 0.35) !important;
        }
        div[data-testid="stChatInput"] textarea,
        .stChatInput textarea {
            color: #f8fafc !important;
        }
        div[data-testid="stChatInput"] textarea::placeholder,
        .stChatInput textarea::placeholder {
            color: #94a3b8 !important;
        }
        div[data-testid="stChatInput"]:has(textarea:disabled),
        .stChatInput:has(textarea:disabled) {
            border-color: #3b82f6 !important;
            background-color: #1e293b !important;
            box-shadow: 0 4px 22px rgba(59, 130, 246, 0.32), 0 0 0 1px rgba(59, 130, 246, 0.35) !important;
            opacity: 1 !important;
        }
    }

    [data-theme="dark"] div[data-testid="stChatInput"],
    .stApp[data-theme="dark"] div[data-testid="stChatInput"] {
        border-color: #3b82f6 !important;
        background-color: #1e293b !important;
        box-shadow: 0 4px 22px rgba(59, 130, 246, 0.32), 0 0 0 1px rgba(59, 130, 246, 0.35) !important;
    }
    [data-theme="dark"] div[data-testid="stChatInput"]:focus-within,
    .stApp[data-theme="dark"] div[data-testid="stChatInput"]:focus-within {
        border-color: #60a5fa !important;
        box-shadow: 0 8px 32px rgba(59, 130, 246, 0.42), 0 0 0 3.5px rgba(59, 130, 246, 0.35) !important;
    }
    [data-theme="dark"] div[data-testid="stChatInput"] textarea,
    .stApp[data-theme="dark"] div[data-testid="stChatInput"] textarea {
        color: #f8fafc !important;
    }
    [data-theme="dark"] div[data-testid="stChatInput"] textarea::placeholder,
    .stApp[data-theme="dark"] div[data-testid="stChatInput"] textarea::placeholder {
        color: #94a3b8 !important;
    }
    [data-theme="dark"] div[data-testid="stChatInput"]:has(textarea:disabled),
    .stApp[data-theme="dark"] div[data-testid="stChatInput"]:has(textarea:disabled) {
        border-color: #3b82f6 !important;
        background-color: #1e293b !important;
        box-shadow: 0 4px 22px rgba(59, 130, 246, 0.32), 0 0 0 1px rgba(59, 130, 246, 0.35) !important;
        opacity: 1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Helper function to index a document payload into ChromaDB
def index_file_data(file_bytes: bytes, filename: str, c_size: int, c_overlap: int, dist_thresh: float, t_k: int, key: str, url: str, mdl: str) -> bool:
    try:
        with st.spinner(f"Extracting text and indexing '{filename}'..."):
            doc = load_document(file_bytes, filename=filename)
            chunks = chunk_document(doc, chunk_size=c_size, chunk_overlap=c_overlap)

            if st.session_state.embedding_service is None:
                st.session_state.embedding_service = get_shared_embedding_service(settings.embedding_model)
            if st.session_state.vector_store is None:
                st.session_state.vector_store = get_shared_vector_store(settings.chroma_persist_directory)

            retriever = SemanticRetriever(
                vector_store=st.session_state.vector_store,
                embedding_service=st.session_state.embedding_service,
                distance_threshold=dist_thresh,
                top_k=t_k,
            )
            retriever.index_document(doc, chunks, force_reindex=True)

            active_api_key = key.strip() if key else settings.active_api_key
            qa_engine = QAEngine(
                retriever=retriever,
                api_key=active_api_key,
                base_url=url.strip() if url else settings.active_base_url,
                model=mdl.strip() if mdl else settings.active_model,
                distance_threshold=dist_thresh,
            )

            st.session_state.qa_engine = qa_engine
            st.session_state.current_doc_info = {
                "filename": doc.filename,
                "document_id": doc.document_id,
                "file_type": doc.file_type.upper(),
                "pages": len(doc.pages),
                "chunks": len(chunks),
                "characters": doc.total_characters,
                "full_text": doc.full_text,
            }
            st.session_state.sample_selected = None
            st.session_state.sample_filename = None
            st.success(f"Indexed '{doc.filename}' ({len(chunks)} chunks)!")
            return True
    except EmptyFileError as e:
        st.error(f"Empty Document: {str(e)}")
    except FileTooLargeError as e:
        st.error(f"File Too Large: {str(e)}")
    except UnsupportedFileTypeError as e:
        st.error(f"Unsupported File: {str(e)}")
    except UnreadableFileError as e:
        st.error(f"Unreadable File: {str(e)}")
    except NoExtractableTextError as e:
        st.error(f"Text Extraction Failed: {str(e)}")
    except Exception as e:
        st.error(f"Processing Error: {str(e)}")
    return False


# Default retrieval & model hyperparameters
api_key_input = settings.active_api_key if settings.has_api_key else ""
base_url_input = settings.active_base_url
model_input = settings.active_model
use_llm = True
top_k = settings.top_k
distance_threshold = settings.retrieval_distance_threshold
chunk_size = 800
chunk_overlap = 150


# ==============================================================================
# SIDEBAR: DOCUMENT INGESTION, ADVANCED SETTINGS & SESSION STORAGE
# ==============================================================================
with st.sidebar:
    st.markdown("## 📂 Document Workspace")
    st.caption("Upload your own document or select a demo to start.")

    st.markdown("#### 1️⃣ Choose Document")
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <span style="font-size: 0.82rem; font-weight: 600; color: #334155;">Upload Document</span>
            <span style="background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; font-size: 0.70rem; font-weight: 700; padding: 2px 7px; border-radius: 9999px;">
                Max Limit: {settings.max_file_size_mb} MB
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        f"Upload a TXT or PDF file (Max: {settings.max_file_size_mb} MB):",
        type=["txt", "pdf"],
        help=f"Upload any text file or text-based PDF up to {settings.max_file_size_mb} MB to index into ChromaDB.",
        label_visibility="collapsed",
    )
    st.markdown(
        f"<div style='font-size: 0.73rem; color: #64748b; margin-top: -6px; margin-bottom: 10px; display: flex; align-items: center; gap: 4px;'>"
        f"<span>🔒 Max upload limit: <b>{settings.max_file_size_mb} MB</b></span> • <span>Formats: <b>.txt, .pdf</b></span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='font-size: 0.82rem; font-weight: 600; color: #475569; margin: 6px 0 3px 0;'>Or select an instant demo:</div>", unsafe_allow_html=True)
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        if st.button("🏢 Datavex", use_container_width=True, help="Load Datavex corporate profile"):
            sample_path = Path("data/datavex_profile.txt")
            if sample_path.exists():
                st.session_state.sample_selected = sample_path.read_bytes()
                st.session_state.sample_filename = "datavex_profile.txt"
                st.toast("Selected 'datavex_profile.txt'. Click Index to load.", icon="📋")
                st.rerun()
            else:
                st.error("data/datavex_profile.txt not found.")
    with col_sb2:
        if st.button("☀️ Solar Tech", use_container_width=True, help="Load Solar Tech research report"):
            sample_path = Path("sample_docs/sample_report.txt")
            if sample_path.exists():
                st.session_state.sample_selected = sample_path.read_bytes()
                st.session_state.sample_filename = "sample_report.txt"
                st.toast("Selected 'sample_report.txt'. Click Index to load.", icon="📋")
                st.rerun()
            else:
                st.error("sample_docs/sample_report.txt not found.")

    file_bytes_to_process = None
    filename_to_process = None

    if uploaded_file is not None:
        raw_uploaded_bytes = uploaded_file.getvalue()
        file_size_mb = len(raw_uploaded_bytes) / (1024 * 1024)
        if len(raw_uploaded_bytes) > settings.max_file_size_bytes:
            st.error(
                f"❌ File '{uploaded_file.name}' ({file_size_mb:.2f} MB) exceeds the maximum allowed limit of "
                f"{settings.max_file_size_mb} MB. Please upload a smaller file or split it into sections."
            )
            file_bytes_to_process = None
            filename_to_process = None
        else:
            file_bytes_to_process = raw_uploaded_bytes
            filename_to_process = uploaded_file.name
    elif st.session_state.sample_selected is not None:
        file_bytes_to_process = st.session_state.sample_selected
        filename_to_process = st.session_state.sample_filename

    # Staged document notice and index button (Step 2 - immediately visible!)
    is_new_staged = file_bytes_to_process is not None and (
        st.session_state.current_doc_info is None
        or st.session_state.current_doc_info.get("filename") != filename_to_process
    )
    if is_new_staged:
        st_box_html = (
            '<div style="background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 10px 12px; margin: 12px 0 8px 0;">'
            '<div style="font-size: 0.70rem; font-weight: 700; color: #1d4ed8; text-transform: uppercase;">Ready to Index</div>'
            f'<div style="font-size: 0.86rem; font-weight: 600; color: #1e40af; word-break: break-all;">📄 {html.escape(filename_to_process)}</div>'
            '</div>'
        )
        st.markdown(st_box_html, unsafe_allow_html=True)
        if st.button(
            f"🚀 Index '{filename_to_process}' Now",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.is_processing,
        ):
            if index_file_data(
                file_bytes_to_process,
                filename_to_process,
                chunk_size,
                chunk_overlap,
                distance_threshold,
                top_k,
                api_key_input,
                base_url_input,
                model_input,
            ):
                st.rerun()

    # Active Document Status Card
    if st.session_state.current_doc_info is not None:
        info = st.session_state.current_doc_info
        active_sb_html = (
            '<div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 12px; margin: 12px 0 8px 0;">'
            '<div style="font-size: 0.70rem; font-weight: 700; color: #166534; text-transform: uppercase;">✅ Active Indexed Document</div>'
            f'<div style="font-weight: 700; font-size: 0.90rem; color: #14532d; word-break: break-all; margin: 4px 0 6px 0;">📑 {html.escape(info["filename"])}</div>'
            '<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 4px; font-size: 0.76rem; color: #166534;">'
            f'<div><b>Format:</b> {info["file_type"]}</div>'
            f'<div><b>Pages:</b> {info["pages"]}</div>'
            f'<div><b>Chunks:</b> {info["chunks"]}</div>'
            f'<div><b>Chars:</b> {info["characters"]:,}</div>'
            '</div>'
            '</div>'
        )
        st.markdown(active_sb_html, unsafe_allow_html=True)

        col_sw, col_vt = st.columns(2)
        with col_sw:
            if st.button("🔄 Switch Doc", use_container_width=True):
                st.session_state.current_doc_info = None
                st.session_state.qa_engine = None
                st.session_state.sample_selected = None
                st.session_state.sample_filename = None
                st.rerun()
        with col_vt:
            with st.popover("👁️ View Text", use_container_width=True):
                st.markdown(f"**Full Extracted Text ({info['filename']}):**")
                st.text_area("Extracted Content", value=info["full_text"], height=350, disabled=True, label_visibility="collapsed")

    # Step 3: Advanced Settings (placed neatly inside an expander so it doesn't block the user)
    st.markdown("---")
    with st.expander("⚙️ Advanced Retrieval Settings (Optional)", expanded=False):
        st.markdown("**LLM API Configuration**")
        api_key_input = st.text_input(
            "API Key",
            value=api_key_input,
            type="password",
            placeholder="Configured in .env or enter here...",
            help="API key for high-speed LLM inference. Stored securely in session.",
        )
        base_url_input = st.text_input(
            "API Base URL",
            value=base_url_input,
            help="Target endpoint base URL.",
        )
        model_input = st.text_input(
            "Model Name",
            value=model_input,
            help="Target chat model identifier.",
        )

        st.markdown("---")
        st.markdown("**Answer Synthesis Engine**")
        generation_mode = st.radio(
            "Generation Mode",
            ["Natural Language Generation", "Local Extractive Fallback (Zero-Cost / Offline)"],
            help="Select Cloud LLM for natural-language generation, or local extractive fallback if offline.",
        )
        use_llm = generation_mode.startswith("Natural Language")

        st.markdown("---")
        st.markdown("**Retrieval Hyperparameters**")
        top_k = st.slider(
            "Top Passages (Top-K)",
            min_value=1,
            max_value=6,
            value=top_k,
            help="Number of nearest ChromaDB passages retrieved.",
        )
        distance_threshold = st.slider(
            "Distance Cutoff Threshold",
            min_value=0.30,
            max_value=0.95,
            value=distance_threshold,
            step=0.01,
            help="Maximum cosine distance cutoff (smaller = stricter relevance).",
        )
        chunk_size = st.number_input(
            "Chunk Size (characters)",
            min_value=100,
            max_value=1500,
            value=chunk_size,
            step=50,
            help="Calibrated for the 256-token limit of all-MiniLM-L6-v2.",
        )
        chunk_overlap = st.number_input(
            "Chunk Overlap (characters)",
            min_value=0,
            max_value=300,
            value=chunk_overlap,
            step=20,
            help="Character overlap between consecutive chunks.",
        )

        st.markdown("---")
        st.caption(f"**Embedding Model:** `{settings.embedding_model}` (Local CPU)")
        st.caption(f"**Vector Database:** ChromaDB (`{settings.chroma_persist_directory}`)")

    # Step 4: Conversation History & Export
    st.markdown("---")
    st.markdown("### 💾 Conversation History")
    msg_count = len(st.session_state.messages)
    turns_count = msg_count // 2
    st.caption(f"{msg_count} messages ({turns_count} Q&A turns) in history.")

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        if st.button("🗑️ Clear Chat", use_container_width=True, disabled=(msg_count == 0)):
            st.session_state.messages = []
            if CONVERSATION_HISTORY_FILE.exists():
                try:
                    CONVERSATION_HISTORY_FILE.unlink()
                except Exception:
                    pass
            st.rerun()

    with col_c2:
        md_export = export_conversation_markdown(st.session_state.messages, st.session_state.current_doc_info)
        st.download_button(
            label="📄 Markdown",
            data=md_export,
            file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True,
            disabled=(msg_count == 0),
        )

    json_export = json.dumps(st.session_state.messages, indent=2)
    st.download_button(
        label="💾 Export JSON",
        data=json_export,
        file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True,
        disabled=(msg_count == 0),
    )


# ==============================================================================
# MAIN WORKSPACE: COMPACT HEADER & PRIMARY CHAT INTERFACE
# ==============================================================================

# Unified Header Box with active document indicator
if st.session_state.current_doc_info:
    info = st.session_state.current_doc_info
    safe_fn = html.escape(info['filename'])
    doc_status_html = (
        '<div class="header-status-card active">'
        '<div class="status-label">🟢 Active Document</div>'
        f'<div class="status-filename" title="{safe_fn}">{safe_fn}</div>'
        f'<div class="status-chunks">{info["chunks"]} chunks indexed</div>'
        '</div>'
    )
else:
    doc_status_html = (
        '<div class="header-status-card empty">'
        '<div class="status-label">⚪ No Document</div>'
        '<div class="status-filename">Upload file in sidebar</div>'
        '<div class="status-chunks">TXT or PDF accepted</div>'
        '</div>'
    )

header_box_html = (
    '<div class="header-box">'
    '<div class="header-left">'
    '<div class="header-title-row">'
    '<span style="font-size: 1.45rem;">📄</span>'
    '<span class="header-title-text">Document Q&A</span>'
    '<span class="header-badge-tag">RAG Assistant</span>'
    '</div>'
    '<p class="header-desc">'
    'Answers grounded strictly in document evidence with exact citations and zero hallucinations.'
    '</p>'
    '</div>'
    f'<div class="header-right">{doc_status_html}</div>'
    '</div>'
)

st.markdown(header_box_html, unsafe_allow_html=True)


# Empty State with Interactive Suggested Questions
if not st.session_state.messages and not st.session_state.pending_chat_prompt:
    welcome_box_html = (
        '<div style="padding: 12px 18px; border-radius: 10px; background: #ffffff; border: 1px solid #e2e8f0; margin: 6px 0 14px 0;">'
        '<div style="display: flex; align-items: center; gap: 8px;">'
        '<span style="font-size: 1.25rem;">🤖</span>'
        '<span style="font-weight: 700; font-size: 0.96rem; color: #0f172a;">Welcome to Document Q&A</span>'
        '</div>'
        '<p style="color: #64748b; font-size: 0.82rem; margin: 4px 0 0 0; line-height: 1.45;">'
        'Ask questions about your uploaded document. Answers are strictly grounded in vector passages with zero hallucination.'
        '</p>'
        '</div>'
    )
    st.markdown(welcome_box_html, unsafe_allow_html=True)

    if st.session_state.current_doc_info:
        st.markdown("<p style='font-size: 0.84rem; font-weight: 600; color: #475569; margin-bottom: 6px;'>💡 Try asking an example question:</p>", unsafe_allow_html=True)
        col_q1, col_q2 = st.columns(2)
        with col_q1:
            if st.button("🏢 When was the company founded?", use_container_width=True):
                st.session_state.pending_chat_prompt = "When was the company founded?"
                st.rerun()
            if st.button("👥 How many employees work there?", use_container_width=True):
                st.session_state.pending_chat_prompt = "How many employees are there?"
                st.rerun()
        with col_q2:
            if st.button("💼 What services are offered?", use_container_width=True):
                st.session_state.pending_chat_prompt = "What services does the company offer?"
                st.rerun()
            if st.button("🛡️ Who is the CEO? (Negative Guardrail Test)", use_container_width=True):
                st.session_state.pending_chat_prompt = "Who is the CEO?"
                st.rerun()
    else:
        empty_picker_html = (
            '<div style="background: #f8fafc; border: 1.5px dashed #cbd5e1; border-radius: 10px; padding: 16px 20px; text-align: center; margin-top: 6px;">'
            '<div style="font-weight: 700; font-size: 0.95rem; color: #0f172a; margin-bottom: 4px;">📂 Choose a Document to Get Started</div>'
            '<p style="font-size: 0.84rem; color: #64748b; margin: 0 0 12px 0;">'
            'Pick an instant demo below to start asking questions immediately, or open the <b>sidebar (top-left ☰)</b> to upload your own TXT or PDF file.'
            '</p>'
            '</div>'
        )
        st.markdown(empty_picker_html, unsafe_allow_html=True)
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            if st.button("🏢 Quick Start with Datavex Profile", use_container_width=True, type="primary"):
                sample_path = Path("data/datavex_profile.txt")
                if sample_path.exists():
                    index_file_data(
                        sample_path.read_bytes(),
                        "datavex_profile.txt",
                        chunk_size,
                        chunk_overlap,
                        distance_threshold,
                        top_k,
                        api_key_input,
                        base_url_input,
                        model_input,
                    )
                    st.rerun()
        with col_m2:
            if st.button("☀️ Quick Start with Solar Tech Report", use_container_width=True):
                sample_path = Path("sample_docs/sample_report.txt")
                if sample_path.exists():
                    index_file_data(
                        sample_path.read_bytes(),
                        "sample_report.txt",
                        chunk_size,
                        chunk_overlap,
                        distance_threshold,
                        top_k,
                        api_key_input,
                        base_url_input,
                        model_input,
                    )
                    st.rerun()


# Render Conversation Messages
for idx, msg in enumerate(st.session_state.messages):
    role = msg.get("role", "user")

    if role == "user":
        with st.chat_message("user"):
            st.markdown(msg.get("content", ""))
            if msg.get("timestamp"):
                st.caption(f"🕒 {msg.get('timestamp')}")

    elif role == "assistant":
        with st.chat_message("assistant"):
            resp_data = msg.get("response_data", {})
            status = resp_data.get("status", "success")
            is_found = resp_data.get("is_found", False)
            content = msg.get("content", "")

            # Case A: Service / API Error
            if status == "api_error":
                st.error(f"**Connection or Execution Failure:** {resp_data.get('error_message', 'Service error')}")
                st.caption("🛡️ Anti-hallucination guardrail active: no answer was fabricated on service error.")
                # Fallback button
                if st.button("⚡ Synthesize with Local Offline Fallback", key=f"retry_fb_{idx}"):
                    if st.session_state.qa_engine and st.session_state.current_doc_info:
                        active_doc_id = st.session_state.current_doc_info["document_id"]
                        fb_res = st.session_state.qa_engine.answer_question(
                            question=resp_data.get("question", msg.get("content", "")),
                            document_id=active_doc_id,
                            top_k=top_k,
                            threshold=distance_threshold,
                            use_llm=False,
                        )
                        msg["content"] = fb_res.answer
                        msg["response_data"] = {
                            "status": fb_res.status,
                            "is_found": fb_res.is_found,
                            "answer": fb_res.answer,
                            "question": fb_res.question,
                            "source_filename": fb_res.source_filename,
                            "page_number": fb_res.page_number,
                            "chunk_id": fb_res.chunk_id,
                            "distance": fb_res.distance,
                            "similarity_score": fb_res.similarity_score,
                            "supporting_passage": fb_res.supporting_passage,
                            "error_message": fb_res.error_message,
                            "citations": fb_res.citations,
                            "retrieved_passages": [
                                {
                                    "source_filename": getattr(p, "source_filename", None) or getattr(p.chunk, "source_filename", ""),
                                    "page_number": getattr(p, "page_number", None) or getattr(p.chunk, "page_number", 1),
                                    "chunk_id": getattr(p, "chunk_id", None) or getattr(p.chunk, "chunk_id", 0),
                                    "distance": getattr(p, "distance", 1.0),
                                    "text": getattr(p, "text", None) or getattr(p.chunk, "text", ""),
                                }
                                for p in (fb_res.retrieved_passages or [])
                            ],
                        }
                        save_persisted_conversation(st.session_state.messages)
                        st.rerun()

            # Case B: Refusal / Evidence Absent
            elif not is_found or content == ANSWER_NOT_FOUND_IN_DOCUMENT:
                st.warning(f"⚠️ **{ANSWER_NOT_FOUND_IN_DOCUMENT}**")
                if msg.get("timestamp"):
                    st.caption(f"🕒 {msg.get('timestamp')}")

            # Case C: Grounded Answer
            else:
                st.markdown(content)

                source_file = html.escape(str(resp_data.get("source_filename", "N/A")))
                page_num = html.escape(str(resp_data.get("page_number", "1")))
                chunk_id = html.escape(str(resp_data.get("chunk_id", "1")))
                dist = resp_data.get("distance", 0.0)
                timestamp = msg.get("timestamp", "")
                time_html = f'<span style="color: #64748b; font-size: 0.78rem; font-weight: 400; margin-left: 6px;">🕒 {html.escape(timestamp)}</span>' if timestamp else ""

                pills_html = (
                    '<div class="cite-pill-row">'
                    f'<span class="cite-pill">📄 <b>File:</b> {source_file}</span>'
                    f'<span class="cite-pill">📖 <b>Page:</b> {page_num}</span>'
                    f'<span class="cite-pill">🧩 <b>Chunk:</b> #{chunk_id}</span>'
                    f'<span class="cite-pill cite-pill-sim">🎯 <b>Dist:</b> {dist:.4f}</span>'
                    f'{time_html}'
                    '</div>'
                )
                st.markdown(pills_html, unsafe_allow_html=True)


# Handle pending prompt execution
if st.session_state.pending_chat_prompt:
    active_prompt = st.session_state.pop("pending_chat_prompt").strip()
    time_now = datetime.now().strftime("%I:%M %p")

    # 1. Render and record user message
    st.session_state.messages.append({"role": "user", "content": active_prompt, "timestamp": time_now})
    with st.chat_message("user"):
        st.markdown(active_prompt)
        st.caption(f"🕒 {time_now}")

    # 2. Synthesize answer with real-time streaming
    with st.chat_message("assistant"):
        qa_engine = st.session_state.qa_engine
        qa_engine.api_key = api_key_input.strip() if api_key_input else settings.active_api_key
        qa_engine.base_url = base_url_input.strip() if base_url_input else settings.active_base_url
        qa_engine.model = model_input.strip() if model_input else settings.active_model
        qa_engine.distance_threshold = distance_threshold

        active_doc_id = st.session_state.current_doc_info["document_id"]
        history_turns = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
            if m.get("content")
        ]

        try:
            if hasattr(qa_engine, "stream_answer_tokens"):
                token_stream = qa_engine.stream_answer_tokens(
                    question=active_prompt,
                    document_id=active_doc_id,
                    top_k=top_k,
                    threshold=distance_threshold,
                    use_llm=use_llm,
                    conversation_history=history_turns,
                )
                streamed_text = st.write_stream(token_stream)
                res = getattr(qa_engine, "last_response", None)
                if res is None:
                    res = qa_engine.answer_question(
                        question=active_prompt,
                        document_id=active_doc_id,
                        top_k=top_k,
                        threshold=distance_threshold,
                        use_llm=use_llm,
                        conversation_history=history_turns,
                    )
            else:
                with st.spinner("Searching ChromaDB evidence & synthesizing grounded answer..."):
                    res = qa_engine.answer_question(
                        question=active_prompt,
                        document_id=active_doc_id,
                        top_k=top_k,
                        threshold=distance_threshold,
                        use_llm=use_llm,
                        conversation_history=history_turns,
                    )
        except Exception as e:
            from src.qa_engine import QAResponse
            res = QAResponse(
                status="api_error",
                is_found=False,
                answer=ANSWER_NOT_FOUND_IN_DOCUMENT,
                question=active_prompt,
                error_message=str(e),
            )

        # Record assistant response so it is preserved across reruns
        asst_msg = {
            "role": "assistant",
            "content": res.answer,
            "timestamp": datetime.now().strftime("%I:%M %p"),
            "response_data": {
                "status": res.status,
                "is_found": res.is_found,
                "answer": res.answer,
                "question": res.question,
                "source_filename": res.source_filename,
                "page_number": res.page_number,
                "chunk_id": res.chunk_id,
                "distance": res.distance,
                "similarity_score": res.similarity_score,
                "supporting_passage": res.supporting_passage,
                "error_message": res.error_message,
                "citations": res.citations,
                "retrieved_passages": [
                    {
                        "source_filename": getattr(p, "source_filename", None) or getattr(p.chunk, "source_filename", ""),
                        "page_number": getattr(p, "page_number", None) or getattr(p.chunk, "page_number", 1),
                        "chunk_id": getattr(p, "chunk_id", None) or getattr(p.chunk, "chunk_id", 0),
                        "distance": getattr(p, "distance", 1.0),
                        "text": getattr(p, "text", None) or getattr(p.chunk, "text", ""),
                    }
                    for p in (res.retrieved_passages or [])
                ],
            },
        }
        st.session_state.messages.append(asst_msg)
        save_persisted_conversation(st.session_state.messages)

    st.rerun()


# ==============================================================================
# CHAT COMPOSER: NATIVE ST.CHAT_INPUT WITH PROPER DISABLED / ACTIVE STATES
# ==============================================================================
doc_is_ready = (st.session_state.qa_engine is not None and st.session_state.current_doc_info is not None)

if doc_is_ready:
    active_filename = st.session_state.current_doc_info["filename"]
    input_placeholder = f"Ask a question about {active_filename} (e.g. When was the company founded?)..."
else:
    input_placeholder = "Upload and index a document in the sidebar to start asking questions..."

is_disabled = st.session_state.is_processing

user_query = st.chat_input(
    placeholder=input_placeholder,
    disabled=is_disabled,
)

if user_query and user_query.strip():
    if not doc_is_ready:
        st.warning("⚠️ Please select or upload a document in the sidebar first to enable question answering.")
    else:
        st.session_state.pending_chat_prompt = user_query.strip()
        st.rerun()
