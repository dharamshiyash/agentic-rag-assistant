import streamlit as st
import sys
import os

# Add the project root to the python path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent import RAGAgent

# Page configuration
st.set_page_config(
    page_title="RAG Agentic Assistant",
    page_icon="🤖",
    layout="centered"
)

def main():
    st.title("🤖 Agentic RAG Assistant")
    st.markdown("""
    Welcome! ask me questions about **AI, Biotechnology, Climate Science, Quantum Computing, Space Exploration, and Sustainable Energy**.
    """)
    
    # Initialize Agent
    if "agent" not in st.session_state:
        try:
            with st.spinner("Initializing Agent and Knowledge Base..."):
                st.session_state.agent = RAGAgent()
            st.success("System Ready!")
        except Exception as e:
            st.error(f"Error initializing agent: {e}")
            st.stop()

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Ask a question..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.agent.answer(prompt)
                st.markdown(response)
        
        # Add assistant message to history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
