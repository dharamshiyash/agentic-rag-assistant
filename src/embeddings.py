import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from src.utils import setup_logger

logger = setup_logger(__name__)

DB_PATH = "faiss_index"

def get_embeddings_model():
    """
    Returns the HuggingFace embeddings model.
    """
    logger.info("Loading embeddings model...")
    # Using a small, efficient model for local use
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

def create_vector_store(chunks):
    """
    Creates a FAISS vector store from document chunks.
    """
    logger.info("Creating vector store...")
    embeddings = get_embeddings_model()
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    logger.info(f"Vector store created with {len(chunks)} documents.")
    return vector_store

def save_vector_store(vector_store, path=DB_PATH):
    """
    Saves the vector store to disk.
    """
    logger.info(f"Saving vector store to {path}...")
    vector_store.save_local(path)
    logger.info("Vector store saved.")

def load_vector_store(path=DB_PATH):
    """
    Loads the vector store from disk.
    If it doesn't exist, returns None.
    """
    if not os.path.exists(path):
        logger.warning(f"Vector store not found at {path}.")
        return None
    
    logger.info(f"Loading vector store from {path}...")
    embeddings = get_embeddings_model()
    vector_store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    logger.info("Vector store loaded.")
    return vector_store

def get_or_create_vector_store(chunks=None):
    """
    Tries to load the vector store; if not found and chunks are provided, creates a new one.
    """
    vector_store = load_vector_store()
    
    if vector_store is None:
        if chunks:
            vector_store = create_vector_store(chunks)
            save_vector_store(vector_store)
        else:
            raise ValueError("Vector store not found and no chunks provided to create one.")
            
    return vector_store
