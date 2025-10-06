from fastapi import FastAPI
from routes.agent_routes import router as agent_router
from routes.run_routes import router as run_router, set_agent as set_run_agent
from routes.health_routes import router as health_router, set_agent_and_memory

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
    app.include_router(agent_router)
    app.include_router(run_router)
    app.include_router(health_router)
    
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
        "main_routes": {
            "GET /": "Service information",
            "GET /docs": "API documentation",
            "GET /redoc": "Alternative API docs"
        }
    }
