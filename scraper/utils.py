"""
Utility functions for logging, error handling, and common operations.
"""
import logging
import sys
from functools import wraps
from typing import Any, Callable
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/scraper.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)


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


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text with ellipsis if needed
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def create_response(success: bool, message: str, data: Any = None) -> dict:
    """
    Create a standardized API response.
    
    Args:
        success: Whether the operation was successful
        message: Response message
        data: Optional data to include
        
    Returns:
        Standardized response dictionary
    """
    response = {
        "success": success,
        "message": message
    }
    if data is not None:
        response["data"] = data
    return response


def log_step(step_name: str):
    """
    Log a processing step.
    
    Args:
        step_name: Name of the step being executed
    """
    logger.info(f"{'='*50}")
    logger.info(f"STEP: {step_name}")
    logger.info(f"{'='*50}")
