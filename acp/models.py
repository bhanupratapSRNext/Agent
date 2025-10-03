from __future__ import annotations
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, AnyUrl, Field

# ---- ACP message format ----
class CitationMetadata(BaseModel):
    kind: Literal["citation"] = "citation"
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    url: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None

class MessagePart(BaseModel):
    name: Optional[str] = None                 # presence implies artifact
    content_type: str                          # e.g., "text/plain", "application/json"
    content: Optional[str] = None
    content_encoding: Optional[Literal["plain","base64"]] = "plain"
    content_url: Optional[AnyUrl] = None
    metadata: Optional[CitationMetadata] = None

    def model_post_init(self, __context: Any) -> None:
        # exactly one of content or content_url
        if (self.content is None) == (self.content_url is None):
            raise ValueError("Provide exactly one of 'content' or 'content_url'.")

class Message(BaseModel):
    role: str  # "user", "agent", or "agent/<name>"
    parts: List[MessagePart]

class Session(BaseModel):
    id: Optional[str] = None
    history: Optional[List[str]] = None
    state: Optional[str] = None

class RunCreateRequest(BaseModel):
    agent_name: str = Field(..., description="ACP agent to run")
    input: List[Message] = Field(..., min_items=1)
    session_id: Optional[str] = None
    session: Optional[Session] = None
    mode: Optional[Literal["sync","async","stream"]] = "sync"

class RunStatus(BaseModel):
    agent_name: str
    session_id: Optional[str] = None
    run_id: str
    status: Literal["created","in-progress","awaiting","cancelling","cancelled","completed","failed"]
    output: Optional[List[Message]] = None
    await_request: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    created_at: str
    finished_at: Optional[str] = None

class AgentManifest(BaseModel):
    name: str
    description: str
    input_content_types: List[str] = ["*/*"]
    output_content_types: List[str] = ["*/*"]
    metadata: Dict[str, Any] = {}
