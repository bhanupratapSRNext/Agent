from fastapi import FastAPI
# from routes.agent_routes import router as agent_router
from routes.run_routes import router as run_router, set_agent as set_run_agent
from routes.health_routes import router as health_router, set_agent_and_memory
from api.configuration_detail import router as pinecone_router
from api.create_connection import router as connection_router
from scraper.scrapper import router as scraper_router  # Add scraper router
from api.user_api import router as user_router  # Add user API router
from api.fetch_configuration import router as fetch_config_router  # Add fetch configuration router

def setup_routes(app: FastAPI, ecommerce_agent, memory):
    """
    Setup all routes with their dependencies
    
    Args:
        app: FastAPI application instance
        ecommerce_agent: The agent instance
        memory: The memory instance
    """
    # Set up route dependencies
    set_run_agent(ecommerce_agent)
    set_agent_and_memory(ecommerce_agent, memory)
    
    # Include all routers
    # app.include_router(agent_router)
    app.include_router(run_router)
    app.include_router(health_router)
    app.include_router(pinecone_router)  # Add Pinecone API router
    app.include_router(connection_router)  # Add Database Connection API router
    app.include_router(scraper_router)  # Add Scraper API router
    app.include_router(user_router)  # Add User API router
    app.include_router(fetch_config_router)  # Add Fetch Configuration API router
    
    return app

def get_route_info():
    """
    Get information about all available routes
    
    Returns:
        dict: Route information
    """
    return {
        "agent_routes": {
            "GET /agents": "List all agents",
            "GET /agents/{agent_name}": "Get specific agent"
        },
        "run_routes": {
            "POST /runs": "Create and execute a run",
            "GET /runs/{run_id}": "Get run status",
            "GET /runs/{run_id}/events": "Get run events",
            "GET /runs": "List all runs"
        },
        "health_routes": {
            "GET /health": "Health check",
            "GET /metrics": "Service metrics"
        },
        "pinecone_routes": {
            "POST /pinecone/create-index": "Create a new Pinecone index",
            "GET /pinecone/list-indexes": "List all Pinecone indexes",
            "DELETE /pinecone/delete-index": "Delete a Pinecone index",
            "GET /pinecone/index-stats/{index_name}": "Get index statistics",
            "POST /pinecone/upload-pdf": "Upload PDF, create embeddings and store in Pinecone"
        },
        "connection_routes": {
            "POST /connections/save": "Save database connection credentials",
            "GET /connections/list": "List all saved connections",
            "GET /connections/{connection_id}": "Get specific connection details",
            "PUT /connections/{connection_id}": "Update connection credentials",
            "DELETE /connections/{connection_id}": "Delete a connection",
            "POST /connections/{connection_id}/test": "Test a database connection"
        },
        "scraper_routes": {
            "GET /fetch/data": "Fetch data using scraper"
        },
        "main_routes": {
            "GET /": "Service information",
            "GET /docs": "API documentation",
            "GET /redoc": "Alternative API docs"
        }
    }
