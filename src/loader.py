import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils import setup_logger

logger = setup_logger(__name__)

def load_documents(data_dir: str):
    """
    Loads all text files from the specified directory.
    """
    logger.info(f"Loading documents from {data_dir}...")
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")
        
    loader = DirectoryLoader(data_dir, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    logger.info(f"Loaded {len(documents)} documents.")
    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Splits documents into smaller chunks.
    """
    logger.info(f"Splitting documents with chunk_size={chunk_size}, overlap={chunk_overlap}...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    
    logger.info(f"Created {len(chunks)} chunks.")
    return chunks

def load_and_split(data_dir: str):
    """
    Convenience function to load and split documents.
    """
    documents = load_documents(data_dir)
    chunks = split_documents(documents)
    return chunks
