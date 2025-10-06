from fastapi import APIRouter
from acp_sdk import RunStatus
from acp_config import ACPConfig

# Create router for health endpoints
router = APIRouter(tags=["health"])

# Import the agent (will be injected)
ecommerce_agent = None
memory = None

def set_agent_and_memory(agent, memory_instance):
    """Set the agent and memory instances for use in routes"""
    global ecommerce_agent, memory
    ecommerce_agent = agent
    memory = memory_instance

@router.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": ecommerce_agent._now_iso(),
        "agents": 1,
        "memory_sessions": len(memory._store),
        "configuration": {
            "rate_limiting": ACPConfig.ENABLE_RATE_LIMITING,
            "authentication": ACPConfig.ENABLE_AUTHENTICATION,
            "logging": ACPConfig.ENABLE_LOGGING
        }
    }

@router.get("/metrics")
def get_metrics():
    """Get service metrics"""
    runs = ecommerce_agent.get_all_runs()
    completed_runs = sum(1 for run in runs.values() if run.status == RunStatus.COMPLETED)
    failed_runs = sum(1 for run in runs.values() if run.status == RunStatus.FAILED)
    
    return {
        "total_runs": len(runs),
        "completed_runs": completed_runs,
        "failed_runs": failed_runs,
        "success_rate": completed_runs / len(runs) if runs else 0,
        "active_sessions": len(memory._store),
        "timestamp": ecommerce_agent._now_iso()
    }
