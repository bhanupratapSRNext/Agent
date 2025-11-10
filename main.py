import os
import uuid
import logging
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Suppress watchfiles INFO logs
logging.getLogger('watchfiles').setLevel(logging.WARNING)

from agent.memory import RollingMemory
from tools.vector_pinecone import VectorRetriever
from tools.sql_postgres import SQLTool

# Import ACP configuration
from acp_config import ACPConfig

# Import the Magentic-One based EcommerceAgent class
from agent.ecommerce_agent import EcommerceAgent

# Import route manager
from routes.route_manager import setup_routes

# Load environment variables
load_dotenv()

# Initialize FastAPI with ACP SDK and Magentic-One
app = FastAPI(
    title="E-commerce Agent Service",
    version="2.0.0",
    description="ACP-compliant e-commerce agent powered by Microsoft's Magentic-One multi-agent orchestrator",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware with configuration
app.add_middleware(
    CORSMiddleware,
    **ACPConfig.get_cors_config()
)

# Initialize your business logic components (kept for backward compatibility)
memory = RollingMemory(window_size=ACPConfig.MEMORY_WINDOW)
retriever = VectorRetriever()
sql_tool = SQLTool()

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
        }
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
