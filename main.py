import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.getLogger('watchfiles').setLevel(logging.WARNING)
logging.getLogger('watchfiles').setLevel(logging.INFO)

from agent.memory import RollingMemory

# Import ACP configuration
from acp_config import ACPConfig

# Import the Magentic-One based EcommerceAgent class
from agent.ecommerce_agent import EcommerceAgent

# Import route manager
from routes.route_manager import setup_routes

# Import scheduler functions
from scheduler.run_scraper import start_scheduler, stop_scheduler

from contextlib import asynccontextmanager

# Load environment variables
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context:
    - Starts the scheduler on startup
    - Stops the scheduler on shutdown
    """
    start_scheduler()
    logging.info("Scheduler started successfully")
    try:
        yield
    finally:
        stop_scheduler()
        logging.info("Scheduler stopped successfully")
# Initialize FastAPI with ACP SDK and Magentic-One
app = FastAPI(
    title="E-commerce Agent Service",
    version="2.0.0",
    description="ACP-compliant e-commerce agent powered by Microsoft's Magentic-One multi-agent orchestrator",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)
# CORS middleware with configuration
app.add_middleware(
    CORSMiddleware,
    **ACPConfig.get_cors_config()
)

# Initialize your business logic components (kept for backward compatibility)
memory = RollingMemory(window_size=ACPConfig.MEMORY_WINDOW)

# Initialize the Magentic-One agent with memory for context-aware validation
ecommerce_agent = EcommerceAgent(memory=memory)


# Setup all routes with dependencies
setup_routes(app, ecommerce_agent, memory)




# Main service endpoint
@app.get("/")
def home():
    """Service information endpoint"""
    return {
        "service": "E-commerce Agent Service (ACP SDK)",
        "version": "2.0.0",
        "acp_compliant": True,
        "agents": [ecommerce_agent.name],
        "capabilities": ["intent_classification", "rag_retrieval", "sql_querying", "response_composition"],
        "configuration": {
            "max_context_length": ACPConfig.AGENT_MAX_CONTEXT_LENGTH,
            "timeout_seconds": ACPConfig.AGENT_TIMEOUT_SECONDS,
            "rate_limiting": ACPConfig.ENABLE_RATE_LIMITING,
            "authentication": ACPConfig.ENABLE_AUTHENTICATION
        },
        "scheduler": "active"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=ACPConfig.SERVER_PORT, 
        reload=ACPConfig.UVICORN_RELOAD,
        log_level=ACPConfig.UVICORN_LOG_LEVEL
    )
