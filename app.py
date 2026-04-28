"""
app.py
------
Streamlit UI for the AI Medical Chatbot (RAG-based).
Features:
  - Chat interface with message history
  - PDF upload for extending knowledge base
  - Source viewer with confidence scores
  - Dark theme support
  - Response time display
  - Sidebar with project info and settings
"""

import os
import time
import json
import streamlit as st
from pathlib import Path

# ── Page config (must be FIRST Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="AI Medical Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Local imports ──────────────────────────────────────────────────────────────
from chatbot import get_response, is_index_ready
from utils.pdf_loader    import load_single_pdf
from utils.text_splitter import split_documents
from utils.embeddings    import get_embedding_model
from utils.vector_store  import (
    load_faiss_index, build_faiss_index, save_faiss_index
)

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_FOLDER      = "data"
FAISS_INDEX_PATH = "faiss_index"
DISCLAIMER       = (
    "⚕️ **Medical Disclaimer**: This chatbot is for **educational use only** "
    "and is **NOT a substitute** for professional medical advice, diagnosis, or treatment. "
    "Always consult a qualified healthcare provider for medical concerns."
)


# ── Custom CSS ─────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    /* Root variables */
    :root {
        --bg-primary:    #0d1117;
        --bg-secondary:  #161b22;
        --bg-card:       #1c2128;
        --accent-blue:   #58a6ff;
        --accent-green:  #3fb950;
        --accent-red:    #f85149;
        --accent-yellow: #e3b341;
        --text-primary:  #e6edf3;
        --text-muted:    #8b949e;
        --border:        #30363d;
        --user-bubble:   #1f4068;
        --bot-bubble:    #1c2128;
    }

    html, body, [class*="css"]  {
        font-family: 'IBM Plex Sans', sans-serif;
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
    }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #0d1b2a 0%, #1a3a5c 50%, #0d1b2a 100%);
        border: 1px solid var(--accent-blue);
        border-radius: 12px;
        padding: 20px 28px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .main-header h1 {
        font-size: 1.6rem;
        font-weight: 600;
        color: var(--accent-blue);
        margin: 0;
    }
    .main-header p {
        font-size: 0.85rem;
        color: var(--text-muted);
        margin: 4px 0 0 0;
    }

    /* Chat messages */
    .chat-message {
        padding: 14px 18px;
        border-radius: 10px;
        margin: 8px 0;
        line-height: 1.6;
        font-size: 0.95rem;
        animation: fadeIn 0.3s ease;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to   { opacity: 1; transform: translateY(0);   }
    }
    .user-message {
        background: var(--user-bubble);
        border-left: 3px solid var(--accent-blue);
        margin-left: 40px;
    }
    .bot-message {
        background: var(--bot-bubble);
        border-left: 3px solid var(--accent-green);
        margin-right: 40px;
        border: 1px solid var(--border);
    }
    .message-label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 6px;
        opacity: 0.6;
    }

    /* Source cards */
    .source-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.82rem;
        font-family: 'IBM Plex Mono', monospace;
    }
    .confidence-bar {
        height: 4px;
        border-radius: 2px;
        margin-top: 6px;
    }

    /* Disclaimer box */
    .disclaimer-box {
        background: rgba(232, 179, 65, 0.08);
        border: 1px solid var(--accent-yellow);
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.85rem;
        color: var(--accent-yellow);
        margin: 12px 0;
    }

    /* Status badges */
    .status-ready   { color: var(--accent-green); font-weight: 600; }
    .status-missing { color: var(--accent-red);   font-weight: 600; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--bg-secondary) !important;
        border-right: 1px solid var(--border);
    }

    /* Input area */
    .stTextInput > div > div > input {
        background-color: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }
    .stButton button {
        background: var(--accent-blue) !important;
        color: #0d1117 !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        transition: all 0.2s !important;
    }
    .stButton button:hover {
        opacity: 0.85 !important;
        transform: translateY(-1px) !important;
    }

    /* Metrics */
    [data-testid="metric-container"] {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 10px;
    }

    /* Scrollable chat area */
    .chat-container {
        max-height: 520px;
        overflow-y: auto;
        padding: 8px;
        scrollbar-width: thin;
        scrollbar-color: var(--border) transparent;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)


# ── Session state initialization ───────────────────────────────────────────────
def init_session():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0
    if "avg_response_time" not in st.session_state:
        st.session_state.avg_response_time = []


# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🏥 Medical Chatbot")
        st.markdown("---")

        # Index status
        ready = is_index_ready(FAISS_INDEX_PATH)
        if ready:
            st.markdown('<span class="status-ready">✅ Index Ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-missing">❌ Index Not Built</span>', unsafe_allow_html=True)
            st.info("Run `python train_index.py` to build the index.")

        st.markdown("---")

        # ── Upload PDF ───────────────────────────────────────────────────
        st.markdown("### 📄 Upload PDF")
        uploaded = st.file_uploader(
            "Add a medical PDF to knowledge base",
            type=["pdf"],
            label_visibility="collapsed"
        )
        if uploaded:
            _handle_pdf_upload(uploaded)

        st.markdown("---")

        # ── Settings ─────────────────────────────────────────────────────
        st.markdown("### ⚙️ Settings")
        top_k = st.slider("Retrieved Chunks (top-k)", 1, 8, 4)
        st.session_state["top_k"] = top_k

        show_sources = st.checkbox("Show Source Chunks", value=True)
        st.session_state["show_sources"] = show_sources

        st.markdown("---")

        # ── Stats ─────────────────────────────────────────────────────────
        st.markdown("### 📊 Session Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Queries", st.session_state.total_queries)
        with col2:
            avg = (sum(st.session_state.avg_response_time) /
                   len(st.session_state.avg_response_time)
                   if st.session_state.avg_response_time else 0)
            st.metric("Avg Time", f"{avg:.1f}s")

        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

        st.markdown("---")

        # ── Project info ──────────────────────────────────────────────────
        st.markdown("### 👩‍💻 Project Info")
        st.markdown("""
        **Team:**
        - SK Salma (23BCE20344)
        - Syed Muskan (23BCE7305)

        **Guide:** Prof. E. Sreenivasa Reddy

        **Institution:** VIT-AP University
        **Department:** SCOPE

        **Stack:** LangChain · FAISS · Flan-T5 · Streamlit
        """)


def _handle_pdf_upload(uploaded_file):
    """Save uploaded PDF, rebuild index."""
    os.makedirs(DATA_FOLDER, exist_ok=True)
    save_path = os.path.join(DATA_FOLDER, uploaded_file.name)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner(f"Processing {uploaded_file.name}..."):
        try:
            doc    = load_single_pdf(save_path)
            chunks = split_documents([doc], chunk_size=800, chunk_overlap=80)
            embed  = get_embedding_model()

            # Try to extend existing index
            existing = load_faiss_index(embed, FAISS_INDEX_PATH)
            if existing:
                from langchain_community.vectorstores import FAISS as _FAISS
                new_store = _FAISS.from_documents(chunks, embed)
                existing.merge_from(new_store)
                save_faiss_index(existing, FAISS_INDEX_PATH)
            else:
                new_store = build_faiss_index(chunks, embed)
                save_faiss_index(new_store, FAISS_INDEX_PATH)

            st.success(f"✅ {uploaded_file.name} added ({len(chunks)} chunks)")
        except Exception as e:
            st.error(f"Error processing PDF: {e}")


# ── Chat rendering ─────────────────────────────────────────────────────────────
def render_chat():
    for msg in st.session_state.chat_history:
        role    = msg["role"]
        content = msg["content"]

        if role == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <div class="message-label">👤 You</div>
                {content}
            </div>
            """, unsafe_allow_html=True)

        elif role == "assistant":
            st.markdown(f"""
            <div class="chat-message bot-message">
                <div class="message-label">🤖 MedBot</div>
                {content}
            </div>
            """, unsafe_allow_html=True)

            # Show metadata if available
            meta = msg.get("meta", {})
            if meta:
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    st.caption(f"⏱ {meta.get('response_time', '?')}s")
                with col2:
                    top_conf = meta.get("top_confidence", 0)
                    color    = "🟢" if top_conf > 0.5 else "🟡" if top_conf > 0.3 else "🔴"
                    st.caption(f"{color} {top_conf:.0%} confidence")
                with col3:
                    sources = meta.get("sources", [])
                    if sources:
                        st.caption(f"📄 {', '.join(sources)}")

                # Source chunks expander
                if st.session_state.get("show_sources") and meta.get("chunks"):
                    with st.expander("🔍 View Retrieved Context"):
                        for i, (chunk, conf) in enumerate(
                            zip(meta["chunks"], meta.get("confidences", []))
                        ):
                            conf_pct = int(conf * 100)
                            bar_color = "#3fb950" if conf_pct > 50 else "#e3b341" if conf_pct > 30 else "#f85149"
                            src = meta["sources"][i] if i < len(meta["sources"]) else "?"
                            st.markdown(f"""
                            <div class="source-card">
                                <strong>Chunk {i+1}</strong> · <em>{src}</em>
                                · Confidence: {conf_pct}%
                                <div class="confidence-bar"
                                     style="width:{conf_pct}%;background:{bar_color}"></div>
                                <br><small>{chunk[:300]}{'...' if len(chunk) > 300 else ''}</small>
                            </div>
                            """, unsafe_allow_html=True)


# ── Main app ───────────────────────────────────────────────────────────────────
def main():
    inject_css()
    init_session()
    render_sidebar()

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <div style="font-size:2.5rem">🏥</div>
        <div>
            <h1>AI Medical Chatbot</h1>
            <p>Powered by RAG · LangChain · FAISS · Flan-T5 · Fully Local</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Disclaimer ─────────────────────────────────────────────────────────
    st.markdown(f'<div class="disclaimer-box">{DISCLAIMER}</div>', unsafe_allow_html=True)

    # ── Index check ────────────────────────────────────────────────────────
    if not is_index_ready(FAISS_INDEX_PATH):
        st.error(
            "⚠️ FAISS index not found. "
            "Please run `python train_index.py` first, "
            "or upload a PDF using the sidebar."
        )

    # ── Suggested questions ────────────────────────────────────────────────
    if not st.session_state.chat_history:
        st.markdown("#### 💡 Try asking:")
        suggestions = [
            "What are the symptoms of malaria?",
            "How can I prevent diabetes?",
            "What is dengue fever?",
            "What foods are good for a fever?",
            "What are the complications of high blood pressure?"
        ]
        cols = st.columns(len(suggestions))
        for col, q in zip(cols, suggestions):
            with col:
                if st.button(q, use_container_width=True):
                    st.session_state["prefill"] = q
                    st.rerun()

    # ── Chat history ────────────────────────────────────────────────────────
    render_chat()

    # ── Input area ──────────────────────────────────────────────────────────
    st.markdown("---")
    col_input, col_btn = st.columns([5, 1])

    prefill = st.session_state.pop("prefill", "")

    with col_input:
        user_input = st.text_input(
            "Your medical question:",
            value=prefill,
            placeholder="e.g. What are the symptoms of malaria?",
            label_visibility="collapsed",
            key="user_input_field"
        )
    with col_btn:
        ask_clicked = st.button("Ask 🔍", use_container_width=True)

    # ── Process query ────────────────────────────────────────────────────────
    if (ask_clicked or user_input) and user_input.strip():
        question = user_input.strip()

        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        if not is_index_ready(FAISS_INDEX_PATH):
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "⚠️ Please build the FAISS index first by running `python train_index.py`."
            })
        else:
            with st.spinner("🔍 Searching medical knowledge base..."):
                top_k = st.session_state.get("top_k", 4)
                result = get_response(
                    question,
                    chat_history=st.session_state.chat_history,
                    top_k=top_k
                )

            # Track stats
            st.session_state.total_queries += 1
            st.session_state.avg_response_time.append(result["response_time"])

            # Build sources list (aligned with chunks)
            sources_expanded = []
            for i, chunk in enumerate(result["chunks"]):
                src = result["sources"][i] if i < len(result["sources"]) else result["sources"][-1] if result["sources"] else "?"
                sources_expanded.append(src)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result["answer"],
                "meta": {
                    "response_time"  : result["response_time"],
                    "top_confidence" : result["confidences"][0] if result["confidences"] else 0,
                    "sources"        : sources_expanded,
                    "chunks"         : result["chunks"],
                    "confidences"    : result["confidences"]
                }
            })

        st.rerun()


if __name__ == "__main__":
    main()
