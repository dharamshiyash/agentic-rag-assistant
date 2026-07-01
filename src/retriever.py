"""
Retrieval module for the Agentic RAG Assistant.
Implements MMR (Maximal Marginal Relevance) retrieval for diverse,
high-quality document retrieval from the FAISS vector store.
"""

from src.utils import setup_logger, log_step

logger = setup_logger(__name__)

# ─── Retrieval Configuration ───
# MMR balances relevance with diversity to avoid redundant results.
SEARCH_TYPE = "mmr"
SEARCH_K = 5           # Number of documents to return
FETCH_K = 20           # Number of candidates to consider for MMR
LAMBDA_MULT = 0.7      # 0 = max diversity, 1 = max relevance


def get_retriever(vector_store):
    """
    Returns a LangChain retriever configured with MMR search.

    MMR (Maximal Marginal Relevance) selects documents that are both
    relevant to the query AND diverse from each other. This prevents
    returning near-duplicate chunks and provides broader context.

    Args:
        vector_store: A FAISS vector store instance.

    Returns:
        A LangChain Retriever object that returns Document objects
        with full metadata intact.
    """
    log_step("Initializing MMR retriever...", "step")
    logger.info(
        f"Creating retriever: type={SEARCH_TYPE}, k={SEARCH_K}, "
        f"fetch_k={FETCH_K}, lambda={LAMBDA_MULT}"
    )

    retriever = vector_store.as_retriever(
        search_type=SEARCH_TYPE,
        search_kwargs={
            "k": SEARCH_K,
            "fetch_k": FETCH_K,
            "lambda_mult": LAMBDA_MULT,
        },
    )

    log_step("Retriever ready.", "success")
    logger.info("Retriever initialized successfully.")
    return retriever


def format_docs_for_prompt(docs) -> str:
    """
    Formats retrieved documents into a structured context string
    for the LLM prompt. Includes source metadata with each chunk
    so the LLM can reference specific documents and sections.

    Args:
        docs: A list of LangChain Document objects with metadata.

    Returns:
        A formatted string containing document content and metadata.
    """
    if not docs:
        return "No relevant documents found."

    formatted_parts = []
    for doc in docs:
        meta = doc.metadata
        source_info = (
            f"[Source: {meta.get('document_name', 'Unknown')} | "
            f"Section: {meta.get('section', 'General')} | "
            f"Chunk: {meta.get('chunk_id', 'N/A')}]"
        )
        formatted_parts.append(f"{source_info}\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted_parts)


def extract_sources(docs) -> list:
    """
    Extracts source metadata from retrieved documents for
    display in the UI as source cards.

    Args:
        docs: A list of LangChain Document objects with metadata.

    Returns:
        A list of dictionaries with source information:
        [{document_name, section, chunk_id, topic}, ...]
    """
    sources = []
    seen = set()

    for doc in docs:
        meta = doc.metadata
        key = (
            meta.get("document_name", ""),
            meta.get("section", ""),
            meta.get("chunk_id", ""),
        )
        if key not in seen:
            seen.add(key)
            sources.append({
                "document_name": meta.get("document_name", "Unknown"),
                "section": meta.get("section", "General"),
                "chunk_id": meta.get("chunk_id", "N/A"),
                "topic": meta.get("topic", "Unknown"),
            })

    return sources
