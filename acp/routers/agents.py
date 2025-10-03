from fastapi import APIRouter
from ..registry import AGENTS

router = APIRouter(prefix="/agents", tags=["acp-agents"])

@router.get("")
def list_agents():
    return {"agents": [a.model_dump() for a in AGENTS.values()]}

@router.get("/{name}/manifest")
def get_manifest(name: str):
    if name not in AGENTS:
        return {"error": f"Agent '{name}' not found"}
    return AGENTS[name].model_dump()
