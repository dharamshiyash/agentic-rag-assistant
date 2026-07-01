"""
Embeddings and vector store module for the Agentic RAG Assistant.
Manages the creation, caching, and loading of the FAISS vector store.
Implements hash-based cache invalidation to avoid unnecessary
embedding regeneration when documents haven't changed.
"""

import os
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from src.utils import setup_logger, log_step, compute_data_hash

logger = setup_logger(__name__)

# ─── Configuration ───
DB_PATH = "faiss_index"
HASH_FILE = os.path.join(DB_PATH, "data_hash.json")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings_model():
    """
    Returns the HuggingFace embeddings model.
    Uses sentence-transformers/all-MiniLM-L6-v2 — a small, efficient
    model well-suited for semantic similarity tasks.

    Returns:
        A HuggingFaceEmbeddings instance.
    """
    log_step("Loading embeddings model...", "step")
    logger.info(f"Loading embeddings model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    log_step("Embeddings model loaded.", "success")
    return embeddings


def create_vector_store(chunks):
    """
    Creates a FAISS vector store from document chunks.

    Args:
        chunks: A list of LangChain Document objects with metadata.

    Returns:
        A FAISS vector store instance.
    """
    log_step("Generating embeddings and creating vector store...", "step")
    logger.info("Creating vector store...")
    embeddings = get_embeddings_model()
    vector_store = FAISS.from_documents(chunks, embeddings)

    log_step(
        f"Vector store created with {len(chunks)} chunks.", "success"
    )
    logger.info(f"Vector store created with {len(chunks)} documents.")
    return vector_store


def save_vector_store(vector_store, path=DB_PATH, data_dir="data"):
    """
    Saves the vector store to disk along with a hash of the source data.
    The hash is used later to determine if embeddings need to be regenerated.

    Args:
        vector_store: The FAISS vector store to save.
        path: Directory path to save the vector store.
        data_dir: Path to the source data directory (for hash computation).
    """
    log_step(f"Saving vector store to {path}...", "step")
    logger.info(f"Saving vector store to {path}...")
    vector_store.save_local(path)

    # Save the data hash alongside the index
    data_hash = compute_data_hash(data_dir)
    hash_data = {"data_hash": data_hash, "data_dir": data_dir}
    with open(HASH_FILE, "w") as f:
        json.dump(hash_data, f)

    log_step("Vector store saved.", "success")
    logger.info("Vector store saved.")


def load_vector_store(path=DB_PATH):
    """
    Loads the vector store from disk.
    Returns None if the store doesn't exist.

    Args:
        path: Directory path to load the vector store from.

    Returns:
        A FAISS vector store instance, or None if not found.
    """
    if not os.path.exists(path):
        logger.warning(f"Vector store not found at {path}.")
        return None

    log_step(f"Loading vector store from {path}...", "step")
    logger.info(f"Loading vector store from {path}...")
    embeddings = get_embeddings_model()
    vector_store = FAISS.load_local(
        path, embeddings, allow_dangerous_deserialization=True
    )
    log_step("Vector store loaded.", "success")
    logger.info("Vector store loaded.")
    return vector_store


def _has_data_changed(data_dir: str) -> bool:
    """
    Checks if the source documents have changed since the last
    vector store was built by comparing file hashes.

    Args:
        data_dir: Path to the source data directory.

    Returns:
        True if data has changed or no hash file exists, False otherwise.
    """
    if not os.path.exists(HASH_FILE):
        return True

    try:
        with open(HASH_FILE, "r") as f:
            stored = json.load(f)
        current_hash = compute_data_hash(data_dir)
        return stored.get("data_hash") != current_hash
    except (json.JSONDecodeError, KeyError):
        return True


def get_or_create_vector_store(chunks=None, data_dir="data", force_rebuild=False):
    """
    Smart vector store loader with hash-based cache invalidation.

    - If the vector store exists and documents haven't changed, reuses it.
    - If documents have changed or force_rebuild is True, rebuilds it.
    - If no vector store exists, creates a new one from chunks.

    Args:
        chunks: Document chunks to create a new store from (optional).
        data_dir: Path to the source data directory.
        force_rebuild: If True, rebuilds regardless of cache state.

    Returns:
        A FAISS vector store instance.
    """
    data_changed = _has_data_changed(data_dir)

    if not force_rebuild and not data_changed and os.path.exists(DB_PATH):
        log_step(
            "Documents unchanged — reusing cached vector store.", "info"
        )
        return load_vector_store()

    if data_changed:
        log_step(
            "Documents changed — rebuilding vector store...", "warning"
        )

    if chunks is None:
        raise ValueError(
            "Vector store needs rebuilding but no chunks were provided."
        )

    vector_store = create_vector_store(chunks)
    save_vector_store(vector_store, data_dir=data_dir)
    return vector_store
