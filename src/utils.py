"""
Utility functions for the Agentic RAG Assistant.
Provides logging, configuration, and helper functions.
"""

import os
import logging
import hashlib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ─── Color codes for terminal output ───
COLORS = {
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
    "GREEN": "\033[92m",
    "BLUE": "\033[94m",
    "CYAN": "\033[96m",
    "YELLOW": "\033[93m",
    "RED": "\033[91m",
    "MAGENTA": "\033[95m",
}


def setup_logger(name=__name__, level=logging.INFO):
    """
    Sets up a logger with the specified name and level.
    Uses a clean, readable format for console output.
    """
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if the logger is already configured
    if not logger.handlers:
        logger.addHandler(handler)

    return logger


def log_step(message: str, style: str = "info"):
    """
    Prints a colored, demo-friendly status message to the console.
    Used for key pipeline stages during demonstrations.

    Args:
        message: The status message to display.
        style: One of 'info', 'success', 'warning', 'error', 'step'.
    """
    style_map = {
        "info": f"{COLORS['CYAN']}ℹ {COLORS['RESET']}",
        "success": f"{COLORS['GREEN']}✓ {COLORS['RESET']}",
        "warning": f"{COLORS['YELLOW']}⚠ {COLORS['RESET']}",
        "error": f"{COLORS['RED']}✗ {COLORS['RESET']}",
        "step": f"{COLORS['BLUE']}→ {COLORS['RESET']}",
    }
    prefix = style_map.get(style, style_map["info"])
    print(f"{prefix}{COLORS['BOLD']}{message}{COLORS['RESET']}")


def get_groq_api_key():
    """
    Retrieves the Groq API key from environment variables.
    Raises an error if the key is missing.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found in environment variables. "
            "Please set it in the .env file."
        )
    return api_key


def compute_data_hash(data_dir: str) -> str:
    """
    Computes a SHA-256 hash of all files in the data directory.
    Used to detect when documents have changed and embeddings
    need to be regenerated.

    Args:
        data_dir: Path to the directory containing knowledge documents.

    Returns:
        A hex digest string representing the combined hash of all files.
    """
    hasher = hashlib.sha256()

    if not os.path.exists(data_dir):
        return ""

    for filename in sorted(os.listdir(data_dir)):
        filepath = os.path.join(data_dir, filename)
        if os.path.isfile(filepath):
            with open(filepath, "rb") as f:
                hasher.update(f.read())

    return hasher.hexdigest()
