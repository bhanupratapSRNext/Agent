import json
import uuid
from typing import Optional, Tuple
from pydantic import ValidationError
from models.schemas import RecommendOut

def run_query(graph, checkpointer, text: str, thread_id: Optional[str] = None) -> Tuple[dict, str]:
    thread_id = thread_id or str(uuid.uuid4())
    result = graph.invoke(
        {"messages": [{"role": "user", "content": text}]},
        config={"configurable": {"thread_id": thread_id}, "checkpoint": checkpointer},
    )
    final = result["messages"][-1].content
    # Normalize various return forms
    if isinstance(final, list) and final and isinstance(final[0], dict) and "text" in final[0]:
        final = final[0]["text"]
    if not isinstance(final, str):
        final = str(final)

    try:
        parsed = RecommendOut.model_validate_json(final)
        return json.loads(parsed.model_dump_json()), thread_id
    except ValidationError:
        return {"rationale": "Model returned non-JSON text.", "recommendations": []}, thread_id
