"""
Streamlit UI for the Agentic RAG Assistant.
Provides a clean, welcoming light-themed chat interface with streaming responses,
suggested follow-up questions, and a sidebar with system information and controls.
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

# ─── Custom CSS (Clean Light Theme) ───
st.markdown("""
<style>
    /* ── Main Theme ── */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ── Headers & Text ── */
    h1, h2, h3, h4, h5, h6, p, span, div {
        color: #0f172a;
    }

    /* ── Chat Messages ── */
    .stChatMessage {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    /* ── Suggested Question Buttons ── */
    .stButton > button {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        color: #1e293b !important;
        padding: 10px 16px;
        font-size: 0.9em;
        font-weight: 500;
        width: 100%;
        text-align: left;
        transition: all 0.2s ease;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }
    .stButton > button:hover {
        border-color: #2563eb;
        color: #2563eb !important;
        background-color: #eff6ff;
        transform: translateY(-1px);
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.1);
    }

    /* ── Sidebar Styling ── */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    .sidebar-header {
        font-size: 1.05em;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 10px;
        padding-bottom: 8px;
        border-bottom: 2px solid #f1f5f9;
    }
    .sidebar-topic {
        background-color: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 8px 12px;
        margin: 6px 0;
        color: #334155;
        font-size: 0.88em;
        font-weight: 500;
        transition: background-color 0.2s ease;
    }
    .sidebar-topic:hover {
        background-color: #e2e8f0;
        color: #0f172a;
    }
    .sidebar-info {
        color: #475569;
        font-size: 0.82em;
        line-height: 1.6;
        margin-top: 4px;
    }

    /* ── Hide Streamlit branding ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ── Section Divider ── */
    .section-divider {
        border-top: 1px solid #e2e8f0;
        margin: 18px 0;
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
            "<b>Model:</b> LLaMA 3.1 8B (Groq)<br>"
            "<b>Embeddings:</b> all-MiniLM-L6-v2<br>"
            "<b>Vector DB:</b> FAISS<br>"
            "<b>Retrieval:</b> MMR (k=5)<br>"
            "<b>Framework:</b> LangChain"
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


def render_suggested_questions(questions):
    """Renders suggested follow-up questions as clickable buttons."""
    if not questions:
        return

    st.markdown("<br>**💡 You may also ask:**", unsafe_allow_html=True)
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
            # Re-render suggestions for assistant messages (sources removed per request)
            if message["role"] == "assistant":
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

            # Get suggested questions (sources removed per request)
            suggested = st.session_state.agent.get_suggested_questions(
                full_response
            )

            # Render suggestions
            render_suggested_questions(suggested)

        # Save assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "suggested_questions": suggested,
        })


if __name__ == "__main__":
    main()
