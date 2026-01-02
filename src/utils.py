import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def setup_logger(name=__name__, level=logging.INFO):
    """
    Sets up a logger with the specified name and level.
    """
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding duplicate handlers if the logger is already configured
    if not logger.handlers:
        logger.addHandler(handler)
        
    return logger

def get_groq_api_key():
    """
    Retrieves the Groq API key from environment variables.
    Raises an error if the key is missing.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in the .env file.")
    return api_key
