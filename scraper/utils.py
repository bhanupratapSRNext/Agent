"""
Utility functions for logging, error handling, and common operations.
"""
import logging
import sys
from pathlib import Path
from functools import wraps
from typing import Any, Callable
import traceback

# Ensure logs directory exists
Path('logs').mkdir(exist_ok=True)

# Get or create the scraper logger (isolated from root logger)
logger = logging.getLogger('scraper')
logger.setLevel(logging.INFO)

# Only configure if not already configured (prevents duplicate handlers on re-import)
if not logger.handlers:
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler (stdout)
    # console_handler = logging.StreamHandler(sys.stdout)
    # console_handler.setLevel(logging.INFO)
    # console_handler.setFormatter(formatter)
    # logger.addHandler(console_handler)

    
    # File handler for scraper.log
    file_handler = logging.FileHandler('logs/scraper.log', mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Prevent propagation to root logger (keeps scraper logs isolated)
    logger.propagate = False


def handle_errors(func: Callable) -> Callable:
    """
    Decorator to handle errors gracefully and log them.
    
    Args:
        func: Function to wrap with error handling
        
    Returns:
        Wrapped function with error handling
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

