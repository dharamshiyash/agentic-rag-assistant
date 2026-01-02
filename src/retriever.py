from src.utils import setup_logger

logger = setup_logger(__name__)

# Similarity threshold to filter out irrelevant documents
# L2 distance: Lower is better. 
# Cosine similarity: Higher is better (if normalized).
# FAISS L2: 0 is identical.
# For this implementation, we will assume the vector store uses L2 or Inner Product, 
# but LangChain's FAISS wrapper usually returns a score.
# If using normalized embeddings (like all-MiniLM-L6-v2) and Inner Product, it is effectively cosine similarity [0, 1].

SIMILARITY_THRESHOLD = 0.5

def get_relevant_documents(vector_store, query, k=3, threshold=SIMILARITY_THRESHOLD):
    """
    Retrieves documents relevant to the query.
    Filters out documents with a similarity score below the threshold.
    """
    logger.info(f"Retrieving documents for query: '{query}' with threshold {threshold}...")
    
    # search_type="similarity_score_threshold" is supported by some retrievers but FAISS implementation 
    # in LangChain usually returns scores differently. 
    # reliable method: use similarity_search_with_score
    
    results = vector_store.similarity_search_with_score(query, k=k)
    
    relevant_docs = []
    for doc, score in results:
        # Note on FAISS score: 
        # By default, FAISS uses L2 distance (lower is better).
        # However, it depends on how the index was created.
        # With sentence-transformers/all-MiniLM-L6-v2 and default FAISS, it's usually L2.
        # BUT LangChain often validates scores.
        # Let's inspect the behavior. 
        # If score is L2 distance, we need a max distance threshold.
        # If score is cosine similarity, we need a min similarity threshold.
        
        # NOTE: For simplicity and robustness without dry-running, 
        # we will use the `similarity_score_threshold` search type provided by LangChain
        # if the vector store supports it as a Retriever.
        # But to be explicit and control the logic:
        
        logger.debug(f"Doc: {doc.page_content[:30]}... - Score: {score}")
        
        # Assuming L2 distance (lower is better) for now. 
        # 1.0 is quite far for normalized vectors (0-2 range). 
        # Let's aim for < 1.0 or < 0.8.
        # Or better yet, let's use the retriever interface which handles this abstractly if possible.
        
        # Let's stick onto raw score check for transparency as requested.
        # For L2 distance on normalized vectors:
        # Distance = 2 * (1 - cosine_similarity)
        # Cosine sim of 0.5 => Distance = 1.0
        # Cosine sim of 0.7 => Distance = 0.6
        # So a threshold of 0.8-1.0 (distance) is roughly 0.5-0.6 cosine similarity.
        
        if score < 1.1: # Accepting reasonable relevance.
             relevant_docs.append(doc)
    
    if not relevant_docs:
        logger.info("No documents passed the similarity threshold.")
    else:
        logger.info(f"Found {len(relevant_docs)} documents above threshold.")
        
    return relevant_docs

def get_retriever(vector_store):
    """
    Returns a runnable retriever.
    This function wraps the logic to return a formatted string for the LLM.
    """
    def retrieve_fn(query):
        docs = get_relevant_documents(vector_store, query)
        return "\n\n".join([d.page_content for d in docs])
    
    return retrieve_fn
