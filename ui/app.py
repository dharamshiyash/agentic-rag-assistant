"""
Streamlit UI for the Agentic RAG Assistant.
Provides a professional chat interface with streaming responses,
source attribution cards, suggested follow-up questions, and
a sidebar with system information and controls.
"""

import streamlit as st
import sys
import os

# Add the project root to the python path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent import RAGAgent

# ─── Page Configuration ───
st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───
st.markdown("""
<style>
    /* ── Main Theme ── */
    .stApp {
        background-color: #0e1117;
    }

    /* ── Chat Messages ── */
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 8px;
    }

    /* ── Source Card ── */
    .source-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #1e2640 100%);
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 6px 0;
        font-size: 0.85em;
        color: #a0aec0;
        transition: border-color 0.2s ease;
    }
    .source-card:hover {
        border-color: #4a6cf7;
    }
    .source-card .source-topic {
        color: #4a6cf7;
        font-weight: 600;
        font-size: 0.95em;
        margin-bottom: 4px;
    }
    .source-card .source-detail {
        color: #718096;
        font-size: 0.85em;
    }

    /* ── Suggested Question Button ── */
    .stButton > button {
        background: linear-gradient(135deg, #1a1f2e 0%, #1e2640 100%);
        border: 1px solid #2d3748;
        border-radius: 8px;
        color: #a0aec0;
        padding: 8px 16px;
        font-size: 0.85em;
        width: 100%;
        text-align: left;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        border-color: #4a6cf7;
        color: #e2e8f0;
        background: linear-gradient(135deg, #1e2640 0%, #243056 100%);
    }

    /* ── Sidebar ── */
    .sidebar-header {
        font-size: 1.1em;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 8px;
        padding-bottom: 8px;
        border-bottom: 1px solid #2d3748;
    }
    .sidebar-topic {
        background: linear-gradient(135deg, #1a1f2e 0%, #1e2640 100%);
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 8px 12px;
        margin: 4px 0;
        color: #a0aec0;
        font-size: 0.9em;
    }
    .sidebar-info {
        color: #718096;
        font-size: 0.8em;
        margin-top: 4px;
    }

    /* ── Sources Header ── */
    .sources-header {
        color: #4a6cf7;
        font-size: 0.9em;
        font-weight: 600;
        margin-top: 12px;
        margin-bottom: 6px;
    }

    /* ── Hide Streamlit branding ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ── Divider ── */
    .section-divider {
        border-top: 1px solid #2d3748;
        margin: 16px 0;
    }
</style>
""", unsafe_allow_html=True)


def render_sidebar():
    """Renders the sidebar with system info, topics, and controls."""
    with st.sidebar:
        st.markdown("## 🧠 Agentic RAG")
        st.markdown(
            '<p class="sidebar-info">'
            "AI assistant powered by Retrieval-Augmented Generation"
            "</p>",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Knowledge Base Topics
        st.markdown(
            '<div class="sidebar-header">📚 Knowledge Base</div>',
            unsafe_allow_html=True,
        )
        topics = [
            "🤖 Artificial Intelligence",
            "🧬 Biotechnology",
            "🌍 Climate Science",
            "⚛️ Quantum Computing",
            "🚀 Space Exploration",
            "⚡ Sustainable Energy",
        ]
        for topic in topics:
            st.markdown(
                f'<div class="sidebar-topic">{topic}</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # System Info
        st.markdown(
            '<div class="sidebar-header">⚙️ System</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="sidebar-info">'
            "Model: LLaMA 3.1 8B (Groq)<br>"
            "Embeddings: all-MiniLM-L6-v2<br>"
            "Vector DB: FAISS<br>"
            "Retrieval: MMR (k=5)<br>"
            "Framework: LangChain"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Controls
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            if "agent" in st.session_state:
                st.session_state.agent.clear_memory()
            st.rerun()

        # File Upload Section
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-header">📄 Upload Documents</div>',
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Add a .txt file to the knowledge base",
            type=["txt"],
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            save_path = os.path.join(
                os.path.dirname(__file__), "..", "data", uploaded_file.name
            )
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Saved: {uploaded_file.name}")
            st.info("Restart the app to include this document in the knowledge base.")


def render_source_cards(sources):
    """Renders source attribution cards for retrieved documents."""
    if not sources:
        return

    st.markdown(
        '<div class="sources-header">📎 Sources</div>',
        unsafe_allow_html=True,
    )

    for source in sources:
        st.markdown(
            f'<div class="source-card">'
            f'<div class="source-topic">{source["topic"]}</div>'
            f'<div class="source-detail">'
            f'Section: {source["section"]} · Chunk: {source["chunk_id"]}'
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


def render_suggested_questions(questions):
    """Renders suggested follow-up questions as clickable buttons."""
    if not questions:
        return

    st.markdown("**💡 You may also ask:**")
    for i, q in enumerate(questions):
        if st.button(q, key=f"suggestion_{len(st.session_state.messages)}_{i}"):
            st.session_state.pending_question = q
            st.rerun()


def main():
    """Main application entry point."""
    render_sidebar()

    # ─── Header ───
    st.markdown("# 🧠 Agentic RAG Assistant")
    st.markdown(
        "Ask me anything about **AI, Biotechnology, Climate Science, "
        "Quantum Computing, Space Exploration,** and **Sustainable Energy**."
    )

    # ─── Initialize Agent ───
    if "agent" not in st.session_state:
        try:
            with st.status("Initializing Knowledge Base...", expanded=True) as status:
                st.write("Loading documents...")
                st.write("Generating embeddings...")
                st.session_state.agent = RAGAgent()
                st.write("System ready!")
                status.update(
                    label="✅ System Ready", state="complete", expanded=False
                )
        except Exception as e:
            st.error(f"Error initializing agent: {e}")
            st.stop()

    # ─── Chat History ───
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Re-render sources and suggestions for assistant messages
            if message["role"] == "assistant":
                if message.get("sources"):
                    render_source_cards(message["sources"])
                if message.get("suggested_questions"):
                    render_suggested_questions(message["suggested_questions"])

    # ─── Handle Pending Suggested Question ───
    prompt = None
    if "pending_question" in st.session_state:
        prompt = st.session_state.pending_question
        del st.session_state.pending_question

    # ─── User Input ───
    if prompt is None:
        prompt = st.chat_input("Ask a question...")

    if prompt:
        # Add and display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and stream response
        with st.chat_message("assistant"):
            # Stream the response token by token
            full_response = st.write_stream(
                st.session_state.agent.stream(prompt)
            )

            # Get sources and suggested questions
            sources = st.session_state.agent.get_sources()
            suggested = st.session_state.agent.get_suggested_questions(
                full_response
            )

            # Render source cards and suggestions
            render_source_cards(sources)
            render_suggested_questions(suggested)

        # Save assistant message with metadata
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": sources,
            "suggested_questions": suggested,
        })


if __name__ == "__main__":
    main()
