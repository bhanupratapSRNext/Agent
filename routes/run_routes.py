import uuid
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional

# Create router for run endpoints
router = APIRouter(prefix="/runs", tags=["runs"])

# Simple request models to avoid ACP SDK validation issues
class MessagePart(BaseModel):
    content_type: str
    content: str
    name: Optional[str] = None

class Message(BaseModel):
    role: str
    parts: List[MessagePart]

class RunCreateRequest(BaseModel):
    agent_name: Optional[str] = "ecommerce-magentic-one"
    input: List[Message]
    session_id: Optional[str] = None
    mode: Optional[str] = "sync"

# Import the agent (will be injected)
ecommerce_agent = None

def set_agent(agent):
    """Set the agent instance for use in routes"""
    global ecommerce_agent
    ecommerce_agent = agent

@router.post("")
async def create_run(request: RunCreateRequest, background_tasks: BackgroundTasks):
    """Create and execute a new run with full ACP compliance"""
    try:
        # Validate request
        if not request.input:
            raise HTTPException(400, "Input messages are required")
        
        if request.input[0].role != "user":
            raise HTTPException(400, "First input message must be from role 'user'")
        
        # Handle agent name mapping
        agent_name = request.agent_name or ecommerce_agent.name
        if agent_name == "router":
            agent_name = "ecommerce-router"
        
        # Generate session ID
        session_id = request.session_id or str(uuid.uuid4())
        
        result = await ecommerce_agent.execute_direct(request, session_id)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Run creation failed: {str(e)}")

@router.get("/{run_id}")
def get_run(run_id: str):
    """Get run status and results"""
    try:
        run = ecommerce_agent.get_run(run_id)
        if not run:
            raise HTTPException(404, f"Run '{run_id}' not found")
        
        return run.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to retrieve run: {str(e)}")

@router.get("/{run_id}/events")
def get_run_events(run_id: str):
    """Get run events and execution details"""
    try:
        run = ecommerce_agent.get_run(run_id)
        if not run:
            raise HTTPException(404, f"Run '{run_id}' not found")
        
        events = []
        if run.status == RunStatus.COMPLETED:
            events.append({
                "timestamp": run.finished_at,
                "event": "run_completed",
                "details": {"status": "success"}
            })
        elif run.status == RunStatus.FAILED:
            events.append({
                "timestamp": run.finished_at,
                "event": "run_failed",
                "details": run.error
            })
        
        return {"events": events, "run_id": run_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to retrieve run events: {str(e)}")

@router.get("")
def list_runs():
    """List all runs (for debugging and monitoring)"""
    runs = ecommerce_agent.get_all_runs()
    return {
        "runs": [run.model_dump() for run in runs.values()],
        "count": len(runs)
    }
