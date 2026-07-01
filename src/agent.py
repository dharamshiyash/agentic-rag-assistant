"""
RAG Agent module for the Agentic RAG Assistant.
Orchestrates the full RAG pipeline: retrieval, prompt construction,
LLM generation, and response post-processing.

Features:
- Conversation memory (sliding window of recent turns)
- Streaming token generation
- Source attribution from retrieved documents
- Suggested follow-up question extraction
"""

import re
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage

from src.utils import setup_logger, log_step, get_groq_api_key
from src.embeddings import get_or_create_vector_store, load_vector_store
from src.loader import load_and_split
from src.retriever import (
    get_retriever,
    format_docs_for_prompt,
    extract_sources,
)
from src.prompts import get_rag_prompt_template

logger = setup_logger(__name__)

# ─── Configuration ───
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0
MEMORY_WINDOW = 5  # Number of conversation turns to keep in memory


class RAGAgent:
    """
    The core RAG Agent that handles question answering with
    retrieval-augmented generation, conversation memory, and streaming.
    """

    def __init__(self):
        log_step("Initializing RAG Agent...", "step")
        logger.info("Initializing RAG Agent...")
        try:
            self.api_key = get_groq_api_key()

            # Initialize LLM
            log_step(f"Loading LLM: {LLM_MODEL}...", "step")
            self.llm = ChatGroq(
                temperature=LLM_TEMPERATURE,
                groq_api_key=self.api_key,
                model_name=LLM_MODEL,
            )
            log_step("LLM ready.", "success")

            # Initialize Vector Store (with cache-aware loading)
            self.vector_store = self._init_vector_store()

            # Initialize Retriever (MMR)
            self.retriever = get_retriever(self.vector_store)

            # Initialize Prompt Template
            self.prompt = get_rag_prompt_template()

            # Conversation memory: stores (HumanMessage, AIMessage) pairs
            self.chat_history = []

            # Build the RAG chain
            self.chain = self._build_chain()

            log_step("RAG Agent initialized successfully.", "success")
            logger.info("RAG Agent initialized successfully.")

        except Exception as e:
            log_step(f"Failed to initialize RAG Agent: {e}", "error")
            logger.error(f"Failed to initialize RAG Agent: {e}")
            raise

    def _init_vector_store(self):
        """
        Initializes the vector store with smart caching.
        Loads existing embeddings if documents haven't changed,
        otherwise rebuilds from the data directory.

        Returns:
            A FAISS vector store instance.
        """
        log_step("Checking vector store...", "step")

        # Try loading existing store first
        try:
            vector_store = get_or_create_vector_store(data_dir="data")
            return vector_store
        except ValueError:
            pass

        # If that fails, load and split documents, then create store
        log_step("Building vector store from documents...", "step")
        chunks = load_and_split("data")
        vector_store = get_or_create_vector_store(
            chunks=chunks, data_dir="data"
        )
        return vector_store

    def _build_chain(self):
        """
        Constructs the RAG chain with retrieval, prompt formatting,
        and LLM generation. The chain accepts a question and chat_history,
        retrieves relevant documents, formats them with metadata, and
        generates a response.

        Returns:
            A LangChain Runnable chain.
        """

        def retrieve_and_format(inputs):
            """Retrieve docs and prepare all inputs for the prompt."""
            question = inputs["question"]
            chat_history = inputs.get("chat_history", [])

            # Retrieve documents
            docs = self.retriever.invoke(question)

            # Store retrieved docs for source extraction later
            self._last_retrieved_docs = docs

            return {
                "context": format_docs_for_prompt(docs),
                "question": question,
                "chat_history": chat_history,
            }

        rag_chain = (
            RunnableLambda(retrieve_and_format)
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain

    def _get_chat_history_messages(self):
        """
        Returns the chat history as a list of LangChain message objects,
        limited to the most recent MEMORY_WINDOW turns.

        Returns:
            A list of HumanMessage and AIMessage objects.
        """
        # Keep only the last N turns
        recent = self.chat_history[-(MEMORY_WINDOW * 2):]
        return recent

    def answer(self, question: str) -> dict:
        """
        Generates an answer for the given question using the RAG pipeline.
        Returns a structured response with the answer, sources, and
        suggested questions.

        Args:
            question: The user's question.

        Returns:
            A dictionary with keys:
            - 'answer': The full response text.
            - 'sources': List of source dictionaries.
            - 'suggested_questions': List of 3 suggested follow-up questions.
        """
        log_step(f"Processing: {question[:80]}...", "step")
        logger.info(f"Processing question: {question}")
        try:
            self._last_retrieved_docs = []

            # Invoke the chain with question and chat history
            response = self.chain.invoke({
                "question": question,
                "chat_history": self._get_chat_history_messages(),
            })

            # Extract sources from retrieved documents
            sources = extract_sources(self._last_retrieved_docs)

            # Extract suggested questions from the response
            answer_text, suggested = self._parse_suggested_questions(response)

            # Update conversation memory
            self.chat_history.append(HumanMessage(content=question))
            self.chat_history.append(AIMessage(content=response))

            log_step("Response generated.", "success")
            logger.info("Response generated successfully.")

            return {
                "answer": answer_text,
                "sources": sources,
                "suggested_questions": suggested,
            }

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            log_step(f"Error: {e}", "error")
            return {
                "answer": "I encountered an error while processing your request. Please try again.",
                "sources": [],
                "suggested_questions": [],
            }

    def stream(self, question: str):
        """
        Streams the response token by token for a modern chatbot experience.
        Yields individual tokens as they are generated.

        This method also updates conversation memory and stores
        retrieved documents for source extraction.

        Args:
            question: The user's question.

        Yields:
            String tokens as they are generated by the LLM.
        """
        log_step(f"Streaming: {question[:80]}...", "step")
        logger.info(f"Streaming response for: {question}")

        self._last_retrieved_docs = []

        try:
            full_response = ""
            for token in self.chain.stream({
                "question": question,
                "chat_history": self._get_chat_history_messages(),
            }):
                full_response += token
                yield token

            # Update conversation memory after streaming completes
            self.chat_history.append(HumanMessage(content=question))
            self.chat_history.append(AIMessage(content=full_response))

            log_step("Streaming complete.", "success")

        except Exception as e:
            logger.error(f"Error during streaming: {e}")
            yield "I encountered an error while processing your request."

    def get_sources(self) -> list:
        """
        Returns the source metadata from the most recently retrieved documents.

        Returns:
            A list of source dictionaries.
        """
        return extract_sources(
            getattr(self, "_last_retrieved_docs", [])
        )

    def get_suggested_questions(self, response_text: str) -> list:
        """
        Extracts suggested follow-up questions from the response text.

        Args:
            response_text: The full response from the LLM.

        Returns:
            A list of suggested question strings.
        """
        _, suggested = self._parse_suggested_questions(response_text)
        return suggested

    def _parse_suggested_questions(self, response: str) -> tuple:
        """
        Parses the LLM response to separate the main answer from
        suggested follow-up questions. Suppresses suggestions if out-of-context.

        Args:
            response: The full response text from the LLM.

        Returns:
            A tuple of (answer_text, list_of_suggested_questions).
        """
        out_of_context_phrases = [
            "out of context for our knowledge base",
            "can't provide an answer as this question is out of context",
            "don't have enough information in my knowledge base",
        ]
        if any(phrase.lower() in response.lower() for phrase in out_of_context_phrases):
            pattern = r"\s*(?:However,\s*I\s*can\s*suggest|\*\*You may also ask:\*\*).*"
            cleaned = re.sub(pattern, "", response, flags=re.DOTALL | re.IGNORECASE).strip()
            return cleaned, []

        suggested = []

        # Look for the "You may also ask:" section
        pattern = r"\*\*You may also ask:\*\*\s*\n((?:\s*[-•]\s*.+\n?)+)"
        match = re.search(pattern, response)

        if match:
            questions_block = match.group(1)
            # Extract individual questions
            questions = re.findall(r"[-•]\s*(.+?)(?:\n|$)", questions_block)
            suggested = [q.strip().strip("?") + "?" for q in questions if q.strip()]

            # Remove the suggested questions section from the answer
            answer_text = response[:match.start()].rstrip()
        else:
            answer_text = response

        return answer_text, suggested[:3]  # Limit to 3 questions

    def clear_memory(self):
        """
        Clears the conversation history.
        Useful when the user wants to start a fresh conversation.
        """
        self.chat_history = []
        log_step("Conversation memory cleared.", "info")
        logger.info("Conversation memory cleared.")
