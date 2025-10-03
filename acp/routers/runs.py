from fastapi import APIRouter, HTTPException
from ..models import RunCreateRequest
from ..store import RUNS, EVENTS
from ..runtime import execute_sync

router = APIRouter(prefix="/runs", tags=["acp-runs"])

@router.post("")
def create_run(req: RunCreateRequest):
    if req.mode not in ("sync", None, "stream", "async"):
        raise HTTPException(400, "Unsupported mode")
    if not req.input or req.input[0].role != "user":
        raise HTTPException(400, "First input message must be from role 'user'")
    part = req.input[0].parts[0]
    if part.content_type != "text/plain":
        raise HTTPException(400, "Only text/plain is supported in this minimal example")

    user_text = part.content or ""
    try:
        status = execute_sync(req.agent_name, user_text, req.session_id or (req.session.id if req.session else None))
    except ValueError as e:
        raise HTTPException(404, str(e))
    return status.model_dump()

@router.get("/{run_id}")
def get_run(run_id: str):
    if run_id not in RUNS:
        raise HTTPException(404, "Run not found")
    return RUNS[run_id].model_dump()

@router.get("/{run_id}/events")
def get_events(run_id: str):
    if run_id not in RUNS:
        raise HTTPException(404, "Run not found")
    return {"events": EVENTS.get(run_id, [])}
