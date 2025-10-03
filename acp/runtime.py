from __future__ import annotations
import uuid,json
from typing import Dict, List, Optional
from .models import Message, MessagePart, RunStatus
from .store import RUNS, EVENTS, now_iso

from agent.memory import RollingMemory
from agent.tools.vector_pinecone import VectorRetriever
from agent.tools.sql_postgres import SQLTool
from agent.router.intent import classify_intent
from agent.router.response_builder import build_answer
import os 
from dotenv import load_dotenv
load_dotenv()

memory = RollingMemory()
retriever = VectorRetriever()
sql_tool = SQLTool()

# MEMORY_WINDOW = int(os.getenv("MEMORY_WINDOW", "6"))
DEFAULT_TOP_K = int(os.getenv("TOP_K", "5"))
DEFAULT_RELEVANCE = float(os.getenv("RELEVANCE_THRESHOLD", "0.55"))

def to_agent_message(text: str, agent_name: str) -> Message:
    return Message(
        role=f"agent/{agent_name}",
        parts=[MessagePart(content_type="text/plain", content=text)]
    )

def run_ecommerce_info(user_id: str, text: str, top_k: int, threshold: float) -> str:
    mem = memory.get(user_id)
    results = retriever.retrieve(text, k=top_k)
    good = [r for r in results if (1.0 - r["score"]) >= threshold or r["score"] >= threshold]
    rag_context = [r["text"] for r in (good or results)]
    if not rag_context:
        return "I couldn't find relevant documents for that."
    return build_answer(text, mem, rag_context, sql_rows=None)

def run_ecommerce_data(user_id: str, text: str, top_k: int, threshold: float) -> str:
    mem = memory.get(user_id)
    out = sql_tool.run(text)
    sql_rows = out.get("rows", [])
    rag_context = None
    if not sql_rows:
        results = retriever.retrieve(text, k=top_k)
        good = [r for r in results if (1.0 - r["score"]) >= threshold or r["score"] >= threshold]
        rag_context = [r["text"] for r in (good or results)]
    if not (sql_rows or rag_context):
        return "No SQL rows found and no relevant docs in vector DB."
    return build_answer(text, mem, rag_context, sql_rows)

def execute_sync(agent_name: Optional[str], user_text: str, session_id: Optional[str]) -> RunStatus:
    
    sid = session_id
    run_id = str(uuid.uuid4())

    # Decide routing mode
    routing_info: Dict[str, str] = {}
    chosen_agent = None

    intent = classify_intent(user_text)
    routing_info["intent"] = intent

    if intent == "Ecommerce_Info":
        chosen_agent = "ecommerce-info"
        answer = run_ecommerce_info(sid, user_text, DEFAULT_TOP_K, DEFAULT_RELEVANCE)
    elif intent == "Ecommerce_Data":
        chosen_agent = "ecommerce-data"
        answer = run_ecommerce_data(sid, user_text, DEFAULT_TOP_K, DEFAULT_RELEVANCE)
    else:
        chosen_agent = "Ecommerce_Info"
        routing_info["note"] = "fallback_default_info"
        answer = run_ecommerce_info(sid, user_text, DEFAULT_TOP_K, DEFAULT_RELEVANCE)

    routing_info["routed_to"] = chosen_agent

    # Update memory
    memory.append(sid, "user", user_text)
    memory.append(sid, "assistant", answer)

    # Build ACP output: main text + a small JSON routing artifact (no schema change)
    output_msgs = [
        to_agent_message(answer, chosen_agent),
        Message(
            role=f"agent/{chosen_agent}",
            parts=[MessagePart(
                name="routing",
                content_type="application/json",
                content=json.dumps(routing_info)
            )]
        )
    ]

    status = RunStatus(
        agent_name=chosen_agent, 
        session_id=sid,
        run_id=run_id,
        status="completed",
        output=output_msgs,
        await_request=None,
        error=None,
        created_at=now_iso(),
        finished_at=now_iso(),
    )
    RUNS[run_id] = status

    EVENTS.setdefault(run_id, []).append({
        "ts": now_iso(),
        "kind": "routing",
        "message": f"Intent-first routing selected {chosen_agent}",
        "details": routing_info,
    })

    return status