from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.utils import setup_logger, get_groq_api_key
from src.embeddings import get_or_create_vector_store, load_vector_store, create_vector_store, save_vector_store
from src.loader import load_and_split
from src.retriever import get_retriever
from src.prompts import get_rag_prompt_template

logger = setup_logger(__name__)

class RAGAgent:
    def __init__(self):
        logger.info("Initializing RAG Agent...")
        try:
            self.api_key = get_groq_api_key()
            
            # Initialize LLM
            # using llama3-70b-8192 for high quality responses
            self.llm = ChatGroq(
                temperature=0,
                groq_api_key=self.api_key,
                model_name="llama-3.1-8b-instant"
            )
            
            # Initialize Vector Store
            self.vector_store = load_vector_store()
            if self.vector_store is None:
                logger.info("Vector store not found. Creating new one from data/...")
                chunks = load_and_split("data")
                self.vector_store = create_vector_store(chunks)
                save_vector_store(self.vector_store)
            
            # Initialize Retriever
            self.retriever = get_retriever(self.vector_store)
            
            # Initialize Chain
            self.prompt = get_rag_prompt_template()
            self.chain = self._build_chain()
            
            logger.info("RAG Agent initialized successfully.")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG Agent: {e}")
            raise

    def _build_chain(self):
        """
        Constructs the RAG chain.
        """
        def format_docs(docs):
            return docs if docs else ""

        rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain

    def answer(self, question: str):
        """
        Generates an answer for the given question.
        """
        logger.info(f"Processing question: {question}")
        try:
            # Invoke the chain
            response = self.chain.invoke(question)
            return response
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I encountered an error while processing your request."
