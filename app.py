"""
app.py
Professional Document Question-Answering Chatbot using RAG.
Features:
- Sleek conversational interface with persistent chat history.
- Local Sentence Transformers embeddings (all-MiniLM-L6-v2) and ChromaDB persistent vector storage.
- Strict anti-hallucination evidence validation (ANSWER_NOT_FOUND_IN_DOCUMENT).
- Natural language generation via Groq LLM API and local extractive fallback.
- Export chat history to Markdown / JSON & auto session persistence.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import inspect
import importlib
import html
import streamlit as st
import streamlit.components.v1 as components

from src.document_loader import (
    load_document,
    DocumentValidationError,
    EmptyFileError,
    UnsupportedFileTypeError,
    UnreadableFileError,
    NoExtractableTextError,
)
from src.text_chunker import chunk_document
from src.embedding_service import EmbeddingService
from src.vector_store import ChromaVectorStore
from src.retriever import SemanticRetriever, TFIDFRetriever

import src.qa_engine
importlib.reload(src.qa_engine)
from src.qa_engine import QAEngine, ANSWER_NOT_FOUND_IN_DOCUMENT
from src.config import settings

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
    page_title="Document Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
        st.session_state.embedding_service = EmbeddingService(settings.embedding_model)
    except Exception:
        st.session_state.embedding_service = None
if "vector_store" not in st.session_state:
    try:
        st.session_state.vector_store = ChromaVectorStore(settings.chroma_persist_directory)
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

# Custom styling for professional conversational layout
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Hero Header */
    .hero-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 22px 26px;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -10px rgba(0, 0, 0, 0.25);
        position: relative;
        overflow: hidden;
    }
    .hero-title {
        font-size: 1.95rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0 0 6px 0;
        letter-spacing: -0.6px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        font-size: 0.92rem;
        color: #94a3b8;
        line-height: 1.5;
        margin: 0 0 14px 0;
    }
    .hero-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
    }
    .hero-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 11px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        background: rgba(255, 255, 255, 0.06);
        color: #cbd5e1;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-pill-active {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border-color: rgba(16, 185, 129, 0.35);
    }

    /* Active Document Card */
    .doc-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 18px 20px;
        color: #f8fafc;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.15);
        margin-top: 14px;
        margin-bottom: 14px;
    }
    .doc-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 10px;
    }
    .doc-card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        word-break: break-all;
    }
    .doc-card-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 8px;
        margin-top: 10px;
    }
    .doc-stat-pill {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 0.80rem;
        color: #cbd5e1;
    }
    .doc-stat-val {
        font-weight: 700;
        color: #38bdf8;
        font-size: 0.95rem;
        display: block;
    }

    /* Chat Layout & Bubble Differentiation */
    .chat-row {
        width: 100%;
        display: flex;
        margin-bottom: 1.25rem;
        position: relative;
        clear: both;
    }

    /* USER CHAT: APPEARING ON THE LEFT SIDE */
    .chat-row-user {
        justify-content: flex-start;
        animation: slideInLeft 0.25s ease-out;
    }
    .user-bubble {
        max-width: 82%;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: #ffffff;
        border-radius: 18px 18px 18px 4px; /* Tail on bottom-left */
        padding: 14px 18px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.16);
        border: 1px solid rgba(255, 255, 255, 0.12);
    }
    .user-bubble-header {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 0.78rem;
        color: #94a3b8;
        margin-bottom: 6px;
        font-weight: 600;
    }
    .user-badge-name {
        color: #38bdf8;
        font-weight: 700;
        letter-spacing: 0.2px;
    }
    .user-bubble-content {
        font-size: 1.02rem;
        font-weight: 500;
        line-height: 1.55;
        color: #f8fafc;
        word-break: break-word;
    }

    /* ASSISTANT REPLY: APPEARING ON THE RIGHT SIDE */
    .chat-row-asst {
        justify-content: flex-end;
        animation: slideInRight 0.25s ease-out;
    }
    .asst-bubble {
        max-width: 88%;
        margin-left: auto;
        border-radius: 18px 18px 4px 18px; /* Tail on bottom-right */
        padding: 18px 22px;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.06);
        background: #ffffff;
        color: #0f172a;
        border: 1.5px solid #e2e8f0;
        position: relative;
    }
    .asst-bubble-grounded {
        border-color: #10b981;
        background: #ffffff;
        box-shadow: 0 6px 24px rgba(16, 185, 129, 0.10);
    }
    .asst-bubble-notfound {
        border-color: #f43f5e;
        background: #fffbfa;
        box-shadow: 0 6px 24px rgba(244, 63, 94, 0.10);
    }
    .asst-bubble-error {
        border-color: #f59e0b;
        background: #fffbeb;
        box-shadow: 0 6px 24px rgba(245, 158, 11, 0.10);
    }
    .asst-bubble-loading {
        border-color: #3b82f6;
        background: #f8fafc;
        box-shadow: 0 4px 18px rgba(59, 130, 246, 0.10);
    }

    .asst-bubble-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(0, 0, 0, 0.06);
    }
    .asst-title-group {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
    }
    .asst-avatar {
        font-size: 1.15rem;
    }
    .asst-name {
        font-weight: 700;
        font-size: 0.90rem;
        color: #1e293b;
    }

    /* Status Badges */
    .status-pill-grounded {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        background-color: #dcfce7;
        color: #15803d;
        border: 1px solid #86efac;
    }
    .status-pill-notfound {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        background-color: #fee2e2;
        color: #b91c1c;
        border: 1px solid #fca5a5;
    }
    .status-pill-error {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        background-color: #fef3c7;
        color: #b45309;
        border: 1px solid #fcd34d;
    }
    .bubble-timestamp {
        font-size: 0.75rem;
        color: #94a3b8;
        white-space: nowrap;
    }

    /* Answer Typography */
    .asst-answer-main {
        font-size: 1.08rem;
        font-weight: 600;
        color: #0f172a;
        line-height: 1.6;
        margin-bottom: 12px;
    }
    .asst-notfound-token {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.05rem;
        font-weight: 700;
        color: #be123c;
        background: #fee2e2;
        padding: 8px 14px;
        border-radius: 8px;
        border: 1px solid #fecdd3;
        display: inline-block;
        margin-bottom: 8px;
    }

    /* Citations & Evidence */
    .citation-badges-container {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .cite-tag {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 4px 9px;
        border-radius: 6px;
        font-size: 0.80rem;
        font-weight: 600;
        background: #f1f5f9;
        color: #334155;
        border: 1px solid #e2e8f0;
    }
    .cite-tag-highlight {
        background: #ecfdf5;
        color: #047857;
        border-color: #a7f3d0;
    }
    .evidence-quote-box {
        background: #f8fafc;
        border-left: 4px solid #10b981;
        border-radius: 4px 8px 8px 4px;
        padding: 10px 14px;
        margin-top: 10px;
        margin-bottom: 8px;
        font-size: 0.88rem;
        color: #334155;
        font-style: italic;
        line-height: 1.5;
    }

    /* Expandable Context Accordion */
    .context-details {
        margin-top: 10px;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        background: #f8fafc;
        overflow: hidden;
    }
    .context-summary {
        cursor: pointer;
        padding: 8px 12px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #475569;
        background: #f1f5f9;
        user-select: none;
        outline: none;
    }
    .context-summary:hover {
        color: #0f172a;
        background: #e2e8f0;
    }
    .context-chunk-item {
        padding: 10px 12px;
        border-top: 1px solid #e2e8f0;
        font-size: 0.82rem;
    }
    .chunk-text-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 8px 10px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #334155;
        white-space: pre-wrap;
        margin-top: 6px;
    }

    /* Thinking Animation */
    .thinking-row {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 0;
    }
    .pulsing-dots {
        display: flex;
        gap: 6px;
    }
    .pulsing-dots span {
        width: 8px;
        height: 8px;
        background-color: #3b82f6;
        border-radius: 50%;
        display: inline-block;
        animation: pulseBounce 1.4s infinite ease-in-out both;
    }
    .pulsing-dots span:nth-child(1) { animation-delay: -0.32s; }
    .pulsing-dots span:nth-child(2) { animation-delay: -0.16s; }
    .pulsing-dots span:nth-child(3) { animation-delay: 0s; }
    .thinking-text {
        font-size: 0.90rem;
        color: #64748b;
        font-weight: 500;
    }

    /* Keyframe Animations */
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-16px); }
        to { opacity: 1; transform: translateX(0); }
    }
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(16px); }
        to { opacity: 1; transform: translateX(0); }
    }
    @keyframes pulseBounce {
        0%, 80%, 100% { transform: scale(0); opacity: 0.4; }
        40% { transform: scale(1.0); opacity: 1; }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_markdown_content(raw_text: str) -> str:
    """Format markdown text (bold, italic, code, bullet lists, citations) into elegant HTML."""
    if not raw_text:
        return ""
    
    import re
    escaped = html.escape(raw_text)
    
    # Inline code: `code`
    escaped = re.sub(
        r"`([^`]+)`",
        r'<code style="background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: 0.88em; color: #0284c7; border: 1px solid #e2e8f0;">\1</code>',
        escaped,
    )
    # Bold: **bold**
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong style='color: #0f172a; font-weight: 700;'>\1</strong>", escaped)
    # Italic: *italic*
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", escaped)
    
    # Citation references like [test.pdf, page 1, chunk 1] or [datavex_profile.txt]
    escaped = re.sub(
        r"\[([a-zA-Z0-9_\-\.\s]+(?:,\s*(?:page|chunk)\s*\d+)*)\]",
        r'<span style="background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; border-radius: 4px; padding: 1px 6px; font-size: 0.80rem; font-weight: 600; display: inline-block; margin: 0 2px;">[\1]</span>',
        escaped,
    )

    lines = escaped.splitlines()
    out = []
    in_ul = False
    
    for line in lines:
        stripped = line.strip()
        if re.match(r"^(?:[\*\-•]|\d+\.)\s+", stripped):
            item_text = re.sub(r"^(?:[\*\-•]|\d+\.)\s+", "", stripped)
            if not in_ul:
                out.append('<ul style="margin: 6px 0 10px 20px; padding: 0; list-style-type: disc;">')
                in_ul = True
            out.append(f'<li style="margin-bottom: 6px; line-height: 1.55;">{item_text}</li>')
        else:
            if in_ul:
                out.append("</ul>")
                in_ul = False
            if stripped:
                out.append(f'<div style="margin-bottom: 8px; line-height: 1.6;">{line}</div>')
            else:
                out.append('<div style="height: 6px;"></div>')
                
    if in_ul:
        out.append("</ul>")
        
    return "\n".join(out)


def clean_html_for_markdown(html_str: str) -> str:
    """Strip leading and trailing whitespace from each line of HTML before passing to st.markdown.

    In CommonMark/Python-Markdown, any line indented with 4 or more leading spaces is parsed
    as an indented code block (<pre><code>). Stripping leading whitespace guarantees that all
    HTML elements (including <details>, <summary>, <div>, etc.) are rendered as native HTML.
    """
    lines = [line.strip() for line in html_str.splitlines() if line.strip()]
    return "\n".join(lines)


def render_user_message_html(content: str, timestamp: str = "") -> str:
    """Render user message bubble on the LEFT side with distinct avatar and styling."""
    escaped_content = html.escape(content).replace("\r\n", "\n").replace("\n", "<br>")
    time_badge = f'<span class="bubble-timestamp">🕒 {html.escape(timestamp)}</span>' if timestamp else ""
    raw_html = f"""
    <div class="chat-row chat-row-user">
        <div class="user-bubble">
            <div class="user-bubble-header">
                <span class="user-badge-icon">🧑‍💻</span>
                <span class="user-badge-name">You</span>
                {time_badge}
            </div>
            <div class="user-bubble-content">{escaped_content}</div>
        </div>
    </div>
    """
    return clean_html_for_markdown(raw_html)


def render_assistant_message_html(ans: str, resp: Dict[str, Any], timestamp: str = "") -> str:
    """Render assistant reply bubble on the RIGHT side with distinct citations and badges."""
    status = resp.get("status", "success")
    is_found = resp.get("is_found", False)
    time_badge = f'<span class="bubble-timestamp">🕒 {html.escape(timestamp)}</span>' if timestamp else ""

    # Case A: API Error
    if status == "api_error":
        err_msg = html.escape(str(resp.get("error_message", "API Request Failed")))
        raw_html = f"""
        <div class="chat-row chat-row-asst">
            <div class="asst-bubble asst-bubble-error">
                <div class="asst-bubble-header">
                    <div class="asst-title-group">
                        <span class="asst-avatar">🤖</span>
                        <span class="asst-name">Document Q&A Assistant</span>
                        <span class="status-pill-error">⚠️ Service Error</span>
                    </div>
                    {time_badge}
                </div>
                <div style="font-weight: 700; color: #b45309; margin-bottom: 6px; font-size: 0.95rem;">Connection or Execution Failure</div>
                <div style="background: rgba(245, 158, 11, 0.08); padding: 10px 14px; border-radius: 8px; color: #92400e; font-size: 0.88rem; font-family: monospace;">
                    {err_msg}
                </div>
                <p style="margin: 8px 0 0 0; color: #78716c; font-size: 0.82rem;">
                    🛡️ Anti-hallucination guardrail active: no answer was fabricated on API error.
                </p>
            </div>
        </div>
        """
        return clean_html_for_markdown(raw_html)

    # Case B: Unsupported / Evidence Absent
    if not is_found or ans == ANSWER_NOT_FOUND_IN_DOCUMENT:
        raw_html = f"""
        <div class="chat-row chat-row-asst">
            <div class="asst-bubble asst-bubble-notfound">
                <div class="asst-bubble-header">
                    <div class="asst-title-group">
                        <span class="asst-avatar">🤖</span>
                        <span class="asst-name">Document Q&A Assistant</span>
                        <span class="status-pill-notfound">🚫 Evidence Absent</span>
                    </div>
                    {time_badge}
                </div>
                <div class="asst-notfound-token">
                    ⚠️ {ANSWER_NOT_FOUND_IN_DOCUMENT}
                </div>
                <p style="margin: 6px 0 0 0; color: #475569; font-size: 0.90rem; line-height: 1.5;">
                    The question contains predicates or requested facts (e.g. absent names, dates, or metrics) that are <b>not verified</b> in the indexed document excerpts.
                    Strict evidence validation intercepted this query to eliminate hallucinations.
                </p>
            </div>
        </div>
        """
        return clean_html_for_markdown(raw_html)

    # Case C: Grounded Answer
    rendered_ans = render_markdown_content(ans)
    source_file = html.escape(str(resp.get("source_filename", "N/A")))
    page_num = html.escape(str(resp.get("page_number", "1")))
    chunk_id = html.escape(str(resp.get("chunk_id", "1")))
    dist = resp.get("distance", 0.0)
    sim = resp.get("similarity_score", 1.0 - dist if dist <= 1.0 else 0.0)

    citations_html = f"""
    <div class="citation-badges-container">
        <span class="cite-tag">📄 <b>File:</b> {source_file}</span>
        <span class="cite-tag">📖 <b>Page:</b> {page_num}</span>
        <span class="cite-tag">🧩 <b>Chunk ID:</b> #{chunk_id}</span>
        <span class="cite-tag cite-tag-highlight">🎯 <b>Cosine Dist:</b> {dist:.4f} (Sim: {sim:.2%})</span>
    </div>
    """

    evidence_html = ""
    supporting_passage = resp.get("supporting_passage")
    if supporting_passage:
        escaped_sup = html.escape(supporting_passage.strip()).replace("\r\n", "\n").replace("\n", "<br>")
        evidence_html = f"""
        <div class="evidence-quote-box">
            <div style="font-weight: 700; color: #047857; font-size: 0.80rem; text-transform: uppercase; margin-bottom: 4px; letter-spacing: 0.5px;">
                💡 Direct Supporting Document Evidence:
            </div>
            "{escaped_sup}"
        </div>
        """

    passages = resp.get("retrieved_passages", [])
    chunks_accordion_html = ""
    if passages:
        chunk_items = []
        for p_idx, p in enumerate(passages, start=1):
            p_file = html.escape(str(p.get("source_filename", "unknown")))
            p_page = html.escape(str(p.get("page_number", 1)))
            p_cid = html.escape(str(p.get("chunk_id", p_idx)))
            p_dist = p.get("distance", 1.0)
            p_text = html.escape(str(p.get("text", ""))).replace("\r\n", "\n").replace("\n", "<br>")
            chunk_items.append(f"""
            <div class="context-chunk-item">
                <div style="display: flex; justify-content: space-between; font-weight: 600; color: #334155;">
                    <span>Rank #{p_idx} • Chunk #{p_cid} • Page {p_page}</span>
                    <span style="color: #0284c7;">Dist: {p_dist:.4f}</span>
                </div>
                <div class="chunk-text-box">{p_text}</div>
            </div>
            """)
        chunks_accordion_html = f"""
        <details class="context-details">
            <summary class="context-summary">🔍 Inspect {len(passages)} ChromaDB Passages Evaluated (Click to Expand)</summary>
            {"".join(chunk_items)}
        </details>
        """

    raw_html = f"""
    <div class="chat-row chat-row-asst">
        <div class="asst-bubble asst-bubble-grounded">
            <div class="asst-bubble-header">
                <div class="asst-title-group">
                    <span class="asst-avatar">🤖</span>
                    <span class="asst-name">Document Q&A Assistant</span>
                    <span class="status-pill-grounded">✨ Grounded in Evidence</span>
                </div>
                {time_badge}
            </div>
            <div class="asst-answer-main">
                {rendered_ans}
            </div>
            {citations_html}
            {evidence_html}
            {chunks_accordion_html}
        </div>
    </div>
    """
    return clean_html_for_markdown(raw_html)


def render_thinking_placeholder_html() -> str:
    """Render animated thinking bubble on the RIGHT side during synthesis."""
    raw_html = """
    <div class="chat-row chat-row-asst">
        <div class="asst-bubble asst-bubble-loading">
            <div class="asst-bubble-header">
                <div class="asst-title-group">
                    <span class="asst-avatar">🤖</span>
                    <span class="asst-name">Document Q&A Assistant</span>
                    <span class="hero-pill hero-pill-active" style="padding: 2px 8px; font-size: 0.72rem;">⚡ Searching Document Evidence...</span>
                </div>
                <span class="bubble-timestamp">Just now</span>
            </div>
            <div class="thinking-row">
                <div class="pulsing-dots">
                    <span></span><span></span><span></span>
                </div>
                <span class="thinking-text">Retrieving semantic vector evidence & synthesizing grounded answer...</span>
            </div>
        </div>
    </div>
    """
    return clean_html_for_markdown(raw_html)


def inject_autoscroll_js():
    """Inject safe client-side JavaScript to smoothly auto-scroll chat to latest message."""
    components.html(
        """
        <script>
        (function() {
            function doScroll() {
                try {
                    const doc = window.parent.document;
                    const anchor = doc.getElementById('chat-bottom-anchor');
                    if (anchor) {
                        anchor.scrollIntoView({ behavior: 'smooth', block: 'end' });
                    }
                    const scrollables = doc.querySelectorAll('div[data-testid="stVerticalBlockBorderWrapper"], div[data-testid="stVerticalBlock"]');
                    scrollables.forEach(function(el) {
                        if (el.scrollHeight > el.clientHeight) {
                            el.scrollTop = el.scrollHeight;
                        }
                    });
                } catch(e) {}
            }
            setTimeout(doScroll, 60);
            setTimeout(doScroll, 220);
            setTimeout(doScroll, 600);
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def render_welcome_card():
    """Render high-end welcome card when chat history is clean."""
    st.markdown(
        clean_html_for_markdown(
            """
            <div class="welcome-card" style="padding: 32px 24px; border-radius: 16px; background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border: 1.5px dashed #cbd5e1; text-align: center; margin-top: 15px;">
                <div style="font-size: 2.8rem; margin-bottom: 12px; filter: drop-shadow(0 4px 8px rgba(0,0,0,0.08));">🤖 💬</div>
                <h3 style="margin: 0 0 8px 0; font-size: 1.35rem; font-weight: 800; color: #0f172a;">Welcome to the Grounded RAG Chatbot</h3>
                <p style="color: #64748b; font-size: 0.95rem; max-width: 620px; margin: 0 auto 16px auto; line-height: 1.6;">
                    Ask any question about your indexed document. Every answer is retrieved from <b>ChromaDB vector chunks</b>, verified with cosine distance thresholds, and validated against strict anti-hallucination guardrails.
                </p>
                <div style="display: flex; justify-content: center; gap: 8px; flex-wrap: wrap;">
                    <span class="cite-tag" style="background: #ffffff;">🔒 Strictly Grounded</span>
                    <span class="cite-tag" style="background: #ffffff;">📑 Exact Provenance Citations</span>
                    <span class="cite-tag" style="background: #ffffff;">💾 Local History Saved</span>
                    <span class="cite-tag" style="background: #ffffff;">⚡ Offline Fallback Mode</span>
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


# Header
st.markdown(
    clean_html_for_markdown(
        """
        <div class="hero-header">
            <div class="hero-title">
                <span>📄 Document Q&A</span>
            </div>
            <div class="hero-subtitle">
                Ask questions and find answers grounded in your documents. Powered by local <b>Sentence Transformers (all-MiniLM-L6-v2)</b>,
                persistent <b>ChromaDB</b> vector store, strict anti-hallucination guardrails, and natural-language answer synthesis.
            </div>
            <div class="hero-pills">
                <span class="hero-pill hero-pill-active">● Vector Store: ChromaDB</span>
                <span class="hero-pill hero-pill-active">● Embeddings: all-MiniLM-L6-v2 (384-d)</span>
                <span class="hero-pill hero-pill-active">● Answer Generation: Active</span>
                <span class="hero-pill">● Anti-Hallucination Guardrail: Calibrated</span>
            </div>
        </div>
        """
    ),
    unsafe_allow_html=True,
)

# Sidebar: Controls and Configuration
with st.sidebar:
    st.header("⚙️ Settings & Configuration")

    st.subheader("⚡ Groq LLM Service")
    api_key_input = st.text_input(
        "Groq API Key",
        value=settings.active_api_key if settings.has_api_key else "",
        type="password",
        placeholder="Configured in .env or enter here...",
        help="Groq API key for high-speed LPU inference. Kept server-side.",
    )
    base_url_input = st.text_input(
        "API Base URL",
        value=settings.active_base_url,
        help="Groq OpenAI-compatible endpoint URL.",
    )
    model_input = st.text_input(
        "Chat Model",
        value=settings.active_model,
        help="Target Groq model identifier.",
    )

    st.markdown("---")
    st.subheader("Synthesis Engine Mode")
    generation_mode = st.radio(
        "Answer Generation Mode",
        ["⚡ Natural Language Generation (Groq LLM)", "⚡ Local Extractive Fallback (Zero-Cost / Offline)"],
        help="Use Groq LLM for natural-language generation, or Local Extractive Synthesis if offline.",
    )
    use_llm = generation_mode.startswith("⚡ Natural Language")

    st.markdown("---")
    st.subheader("Retrieval Hyperparameters")
    top_k = st.slider(
        "Top Passages (Top-K)",
        min_value=1,
        max_value=6,
        value=settings.top_k,
        help="Number of candidate passages to retrieve from ChromaDB.",
    )
    distance_threshold = st.slider(
        "Cosine Distance Threshold",
        min_value=0.30,
        max_value=0.95,
        value=settings.retrieval_distance_threshold,
        step=0.01,
        help="Cosine distance cutoff (smaller = closer). Passages with distance > threshold are rejected.",
    )
    chunk_size = st.number_input(
        "Chunk Size (~tokens)",
        min_value=100,
        max_value=1500,
        value=700,
        step=50,
        help="Approximate tokens per chunk.",
    )
    chunk_overlap = st.number_input(
        "Chunk Overlap (~tokens)",
        min_value=0,
        max_value=300,
        value=100,
        step=20,
        help="Overlap tokens between adjacent chunks.",
    )

    st.markdown("---")
    st.subheader("Local Infrastructure Status")
    st.caption(f"**Embedding Model**: `{settings.embedding_model}` (Local CPU)")
    st.caption(f"**Vector Store**: ChromaDB (Directory: `{settings.chroma_persist_directory}`)")

    st.markdown("---")
    st.subheader("Quick Test Datasets")
    col_sb_d1, col_sb_d2 = st.columns(2)
    with col_sb_d1:
        if st.button("🏢 Datavex Profile", use_container_width=True):
            sample_path = Path("data/datavex_profile.txt")
            if sample_path.exists():
                st.session_state.sample_selected = sample_path.read_bytes()
                st.session_state.sample_filename = "datavex_profile.txt"
                st.success("Loaded Datavex profile into staging! Click 'Index Document' to process.")
            else:
                st.error("data/datavex_profile.txt not found.")
    with col_sb_d2:
        if st.button("☀️ Solar Tech Report", use_container_width=True):
            sample_path = Path("sample_docs/sample_report.txt")
            if sample_path.exists():
                st.session_state.sample_selected = sample_path.read_bytes()
                st.session_state.sample_filename = "sample_report.txt"
                st.success("Loaded Solar Tech Report into staging! Click 'Index Document' to process.")
            else:
                st.error("sample_docs/sample_report.txt not found.")

    st.markdown("---")
    st.subheader("💾 Conversation Storage")
    msg_count = len(st.session_state.messages)
    turns_count = msg_count // 2
    st.caption(f"**History**: {msg_count} messages ({turns_count} Q&A turns) saved.")
    
    col_sb_clear, col_sb_exp = st.columns(2)
    with col_sb_clear:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            if CONVERSATION_HISTORY_FILE.exists():
                try:
                    CONVERSATION_HISTORY_FILE.unlink()
                except Exception:
                    pass
            st.rerun()

    md_export_data = export_conversation_markdown(st.session_state.messages, st.session_state.current_doc_info)
    st.download_button(
        label="📄 Export Markdown",
        data=md_export_data,
        file_name=f"chat_transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        use_container_width=True,
    )
    json_export_data = json.dumps(st.session_state.messages, indent=2)
    st.download_button(
        label="💾 Export JSON",
        data=json_export_data,
        file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True,
    )

    st.markdown("---")
    st.caption("Document Q&A System • Enterprise RAG Edition")

# Main Content Layout: Two Columns (Left = Document Ingestion, Right = Chatbot)
col_doc, col_chat = st.columns([1, 1.8], gap="large")

with col_doc:
    st.subheader("1. Document Ingestion")

    uploaded_file = st.file_uploader(
        "Upload a plain-text (.txt) or text-based PDF (.pdf)",
        type=["txt", "pdf"],
        help="Scanned/image-only PDFs and empty files will be cleanly rejected.",
    )

    # Process either uploaded file or quick sample
    file_bytes_to_process = None
    filename_to_process = None

    if uploaded_file is not None:
        file_bytes_to_process = uploaded_file.getvalue()
        filename_to_process = uploaded_file.name
    elif st.session_state.sample_selected is not None:
        file_bytes_to_process = st.session_state.sample_selected
        filename_to_process = st.session_state.sample_filename
        st.info(f"Staged Sample: **{filename_to_process}** ready for indexing.")

    if file_bytes_to_process is not None:
        btn_label = f"📥 Index '{filename_to_process}' into ChromaDB"
        if st.button(btn_label, type="primary", use_container_width=True, disabled=st.session_state.is_processing):
            try:
                with st.spinner("Extracting text, chunking, and computing local Sentence Transformers embeddings..."):
                    doc = load_document(file_bytes_to_process, filename=filename_to_process)
                    chunks = chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

                    if st.session_state.embedding_service is None:
                        st.session_state.embedding_service = EmbeddingService(settings.embedding_model)
                    if st.session_state.vector_store is None:
                        st.session_state.vector_store = ChromaVectorStore(settings.chroma_persist_directory)

                    retriever = SemanticRetriever(
                        vector_store=st.session_state.vector_store,
                        embedding_service=st.session_state.embedding_service,
                        distance_threshold=distance_threshold,
                        top_k=top_k,
                    )
                    retriever.index_document(doc, chunks, force_reindex=True)

                    active_api_key = api_key_input.strip() if api_key_input else settings.active_api_key
                    qa_engine = QAEngine(
                        retriever=retriever,
                        api_key=active_api_key,
                        base_url=base_url_input.strip() if base_url_input else settings.active_base_url,
                        model=model_input.strip() if model_input else settings.active_model,
                        distance_threshold=distance_threshold,
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
                    st.success(f"Indexed '{doc.filename}' ({len(chunks)} chunks) in ChromaDB!")
                    st.toast(f"Ready to chat with '{doc.filename}'!", icon="🤖")

            except EmptyFileError as e:
                st.error(f"Validation Error: {str(e)}")
            except UnsupportedFileTypeError as e:
                st.error(f"Unsupported File: {str(e)}")
            except UnreadableFileError as e:
                st.error(f"Unreadable File: {str(e)}")
            except NoExtractableTextError as e:
                st.error(f"Extraction Error: {str(e)}")
            except Exception as e:
                st.error(f"Processing Error: {str(e)}")

    # Display Active Document Card with Modern Executive Styling
    if st.session_state.current_doc_info is not None:
        info = st.session_state.current_doc_info
        st.markdown(
            clean_html_for_markdown(
                f"""
                <div class="doc-card">
                    <div class="doc-card-header">
                        <div>
                            <div style="font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.5px; color: #94a3b8; font-weight: 700;">Active Document</div>
                            <h4 class="doc-card-title">📑 {html.escape(info['filename'])}</h4>
                        </div>
                        <span class="hero-pill hero-pill-active" style="padding: 3px 9px; font-size: 0.72rem;">● Indexed</span>
                    </div>
                    <div class="doc-card-grid">
                        <div class="doc-stat-pill">
                            <span class="doc-stat-val">{info['file_type']}</span>
                            Document Format
                        </div>
                        <div class="doc-stat-pill">
                            <span class="doc-stat-val">{info['pages']}</span>
                            Total Pages
                        </div>
                        <div class="doc-stat-pill">
                            <span class="doc-stat-val">{info['chunks']}</span>
                            ChromaDB Chunks
                        </div>
                        <div class="doc-stat-pill">
                            <span class="doc-stat-val">{info['characters']:,}</span>
                            Total Characters
                        </div>
                    </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

        col_reset, col_clear_chat = st.columns([1, 1])
        with col_reset:
            if st.button("🔄 Change Document", use_container_width=True):
                st.session_state.current_doc_info = None
                st.session_state.qa_engine = None
                st.session_state.sample_selected = None
                st.rerun()
        with col_clear_chat:
            if st.button("🧹 Clear Conversation", use_container_width=True):
                st.session_state.messages = []
                if CONVERSATION_HISTORY_FILE.exists():
                    try:
                        CONVERSATION_HISTORY_FILE.unlink()
                    except Exception:
                        pass
                st.rerun()

        with st.expander("👁️ View Extracted Document Content"):
            st.text(info["full_text"])
    else:
        st.info("👈 Upload a document or load the Datavex sample profile to activate the chatbot.")

with col_chat:
    # Status Banner for Chatbot
    if st.session_state.current_doc_info is not None:
        doc_fname = st.session_state.current_doc_info["filename"]
        doc_chunks = st.session_state.current_doc_info["chunks"]
        st.markdown(
            f'<div class="status-pill status-pill-active">🟢 Active Document: <b>{html.escape(doc_fname)}</b> &nbsp;|&nbsp; {doc_chunks} ChromaDB chunks ready for queries</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-pill status-pill-inactive">⚪ Awaiting Document — Please upload and index a document on the left</div>',
            unsafe_allow_html=True,
        )

    # Chat Header with Action Buttons
    chat_header_col1, chat_header_col2 = st.columns([1.5, 1])
    with chat_header_col1:
        st.subheader("2. Conversational Chatbot")
    with chat_header_col2:
        if st.session_state.messages:
            c_export, c_clear = st.columns(2)
            with c_export:
                st.download_button(
                    label="💾 Export",
                    data=export_conversation_markdown(st.session_state.messages, st.session_state.current_doc_info),
                    file_name=f"chat_{datetime.now().strftime('%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True,
                    help="Download entire conversation transcript with citations.",
                )
            with c_clear:
                if st.button("🗑️ Reset", use_container_width=True, help="Clear message history"):
                    st.session_state.messages = []
                    if CONVERSATION_HISTORY_FILE.exists():
                        try:
                            CONVERSATION_HISTORY_FILE.unlink()
                        except Exception:
                            pass
                    st.rerun()

    # CHAT MESSAGES WINDOW (SCROLLABLE CONTAINER - STRICTLY ABOVE INPUT BAR)
    chat_window = st.container(height=650)

    with chat_window:
        if not st.session_state.messages and not st.session_state.get("pending_chat_prompt"):
            render_welcome_card()
        else:
            # Render all existing messages in session state
            for idx, msg in enumerate(st.session_state.messages):
                if msg.get("role") == "user":
                    st.markdown(
                        render_user_message_html(msg.get("content", ""), msg.get("timestamp", "")),
                        unsafe_allow_html=True,
                    )
                elif msg.get("role") == "assistant":
                    resp_data = msg.get("response_data", {})
                    st.markdown(
                        render_assistant_message_html(msg.get("content", ""), resp_data, msg.get("timestamp", "")),
                        unsafe_allow_html=True,
                    )
                    # Inline retry button if API error
                    if resp_data.get("status") == "api_error":
                        retry_key = f"retry_fb_{idx}"
                        if st.button("⚡ Synthesize with Local Offline Fallback", key=retry_key, use_container_width=True):
                            if st.session_state.qa_engine and st.session_state.current_doc_info:
                                active_doc_id = st.session_state.current_doc_info["document_id"]
                                qa_engine = st.session_state.qa_engine
                                fallback_res = qa_engine.answer_question(
                                    question=resp_data.get("question", msg.get("content", "")),
                                    document_id=active_doc_id,
                                    top_k=top_k,
                                    threshold=distance_threshold,
                                    use_llm=False,
                                )
                                msg["content"] = fallback_res.answer
                                msg["response_data"] = {
                                    "status": fallback_res.status,
                                    "is_found": fallback_res.is_found,
                                    "answer": fallback_res.answer,
                                    "question": fallback_res.question,
                                    "source_filename": fallback_res.source_filename,
                                    "page_number": fallback_res.page_number,
                                    "chunk_id": fallback_res.chunk_id,
                                    "distance": fallback_res.distance,
                                    "similarity_score": fallback_res.similarity_score,
                                    "supporting_passage": fallback_res.supporting_passage,
                                    "error_message": fallback_res.error_message,
                                    "citations": fallback_res.citations,
                                    "retrieved_passages": [
                                        {
                                            "source_filename": getattr(p, "source_filename", None) or getattr(p.chunk, "source_filename", ""),
                                            "page_number": getattr(p, "page_number", None) or getattr(p.chunk, "page_number", 1),
                                            "chunk_id": getattr(p, "chunk_id", None) or getattr(p.chunk, "chunk_id", 0),
                                            "distance": getattr(p, "distance", 1.0),
                                            "text": getattr(p, "text", None) or getattr(p.chunk, "text", ""),
                                        }
                                        for p in (fallback_res.retrieved_passages or [])
                                    ],
                                }
                                save_persisted_conversation(st.session_state.messages)
                                st.rerun()

        # Handle newly submitted prompt INSIDE the chat container (above the input bar!)
        if st.session_state.get("pending_chat_prompt"):
            active_q = st.session_state.pop("pending_chat_prompt").strip()
            time_now = datetime.now().strftime("%I:%M %p")

            # 1. Append user message and render immediately on the LEFT
            user_msg = {"role": "user", "content": active_q, "timestamp": time_now}
            st.session_state.messages.append(user_msg)
            st.markdown(render_user_message_html(active_q, time_now), unsafe_allow_html=True)

            # 2. Render thinking animation on the RIGHT
            loading_placeholder = st.empty()
            loading_placeholder.markdown(render_thinking_placeholder_html(), unsafe_allow_html=True)

            # 3. Synchronize QA Engine settings
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

            # 4. Synthesize Answer
            try:
                res = qa_engine.answer_question(
                    question=active_q,
                    document_id=active_doc_id,
                    top_k=top_k,
                    threshold=distance_threshold,
                    use_llm=use_llm,
                    conversation_history=history_turns,
                )
            except TypeError as te:
                if "conversation_history" in str(te):
                    res = qa_engine.answer_question(
                        question=active_q,
                        document_id=active_doc_id,
                        top_k=top_k,
                        threshold=distance_threshold,
                        use_llm=use_llm,
                    )
                else:
                    raise
            except Exception as e:
                from src.qa_engine import QAResponse
                res = QAResponse(
                    status="api_error",
                    is_found=False,
                    answer=ANSWER_NOT_FOUND_IN_DOCUMENT,
                    question=active_q,
                    error_message=str(e),
                )

            # 5. Append Assistant Message
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
            loading_placeholder.empty()
            st.rerun()

        # Anchor tag for auto-scrolling to latest message
        st.markdown('<div id="chat-bottom-anchor"></div>', unsafe_allow_html=True)
        inject_autoscroll_js()

    # INPUT BAR STRICTLY POSITIONED AT THE VERY BOTTOM BELOW THE CHAT CONTAINER!
    chat_prompt = st.chat_input(
        "Ask a question about the document (e.g., When was the company founded?)...",
        disabled=(st.session_state.qa_engine is None),
    )

    if chat_prompt and chat_prompt.strip():
        if st.session_state.qa_engine is None or st.session_state.current_doc_info is None:
            st.warning("Please upload and index a document first to enable question answering.")
        else:
            st.session_state.pending_chat_prompt = chat_prompt.strip()
            st.rerun()
