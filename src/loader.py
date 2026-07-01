"""
Document loading and chunking module for the Agentic RAG Assistant.
Handles loading text documents, splitting them into semantically meaningful
chunks, and enriching each chunk with metadata for source attribution.
"""

import os
import re
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils import setup_logger, log_step

logger = setup_logger(__name__)

# ─── Chunking Configuration ───
# Tuned for retrieval quality with Markdown-structured documents.
# chunk_size=1500 keeps sections intact.
# chunk_overlap=300 preserves context at boundaries.
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300

# Heading-aware separators: prefer splitting at section boundaries.
SEPARATORS = ["\n\n## ", "\n\n### ", "\n\n", "\n", " "]


def load_documents(data_dir: str):
    """
    Loads all .txt files from the specified directory.

    Args:
        data_dir: Path to the directory containing knowledge documents.

    Returns:
        A list of LangChain Document objects.
    """
    log_step("Loading documents...", "step")
    logger.info(f"Loading documents from {data_dir}...")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    loader = DirectoryLoader(data_dir, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()

    log_step(f"Loaded {len(documents)} documents.", "success")
    logger.info(f"Loaded {len(documents)} documents.")
    return documents


def _extract_topic_from_filename(filename: str) -> str:
    """
    Derives a human-readable topic name from a filename.
    Example: 'artificial_intelligence.txt' → 'Artificial Intelligence'

    Args:
        filename: The filename (with or without path).

    Returns:
        A cleaned, title-cased topic name.
    """
    basename = os.path.basename(filename)
    name = os.path.splitext(basename)[0]
    return name.replace("_", " ").title()


def _extract_section_from_content(content: str) -> str:
    """
    Extracts the nearest heading from the beginning of a chunk's content.
    Looks for Markdown headings (## or ###) to determine the section.

    Args:
        content: The text content of a chunk.

    Returns:
        The section heading if found, otherwise 'General'.
    """
    # Search for the first Markdown heading in the chunk
    match = re.search(r"^#{1,3}\s+(.+)$", content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return "General"


def _enrich_metadata(chunks, source_file: str):
    """
    Adds structured metadata to each chunk for source attribution.
    Metadata includes: document_name, topic, section, and chunk_id.

    Args:
        chunks: List of LangChain Document objects (chunks).
        source_file: The source filename for all chunks.

    Returns:
        The same list with enriched metadata on each chunk.
    """
    topic = _extract_topic_from_filename(source_file)

    for i, chunk in enumerate(chunks):
        chunk.metadata["document_name"] = os.path.basename(source_file)
        chunk.metadata["topic"] = topic
        chunk.metadata["section"] = _extract_section_from_content(
            chunk.page_content
        )
        chunk.metadata["chunk_id"] = i + 1

    return chunks


def split_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Splits documents into smaller, semantically meaningful chunks.
    Uses heading-aware separators to avoid splitting sections mid-content.

    Args:
        documents: List of LangChain Document objects.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        A list of enriched Document chunks with metadata.
    """
    log_step("Splitting documents into chunks...", "step")
    logger.info(
        f"Splitting documents with chunk_size={chunk_size}, "
        f"overlap={chunk_overlap}..."
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        add_start_index=True,
    )

    all_chunks = []

    for doc in documents:
        # Split each document individually so we can enrich per-document metadata
        doc_chunks = text_splitter.split_documents([doc])
        source_file = doc.metadata.get("source", "unknown")
        enriched = _enrich_metadata(doc_chunks, source_file)
        all_chunks.extend(enriched)

    log_step(f"Created {len(all_chunks)} chunks with metadata.", "success")
    logger.info(f"Created {len(all_chunks)} chunks.")
    return all_chunks


def load_and_split(data_dir: str):
    """
    Convenience function to load documents and split them into
    enriched chunks in a single call.

    Args:
        data_dir: Path to the directory containing knowledge documents.

    Returns:
        A list of enriched Document chunks ready for embedding.
    """
    documents = load_documents(data_dir)
    chunks = split_documents(documents)
    return chunks
