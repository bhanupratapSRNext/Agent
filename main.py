from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from agent.memory import RollingMemory
from agent.tools.vector_pinecone import VectorRetriever
from agent.tools.sql_postgres import SQLTool
# from agent.tools.mongo_tool import MongoTool
from agent.router.intent import classify_intent
from agent.router.response_builder import build_answer

# Load env
load_dotenv()

MEMORY_WINDOW = int(os.getenv("MEMORY_WINDOW", "6"))
DEFAULT_TOP_K = int(os.getenv("TOP_K", "5"))
DEFAULT_RELEVANCE = float(os.getenv("RELEVANCE_THRESHOLD", "0.55"))

# Initialize FastAPI
app = FastAPI(title="Agent Recommendation Service", version="1.0.0")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tools
memory = RollingMemory(window_size=MEMORY_WINDOW)
retriever = VectorRetriever()
sql_tool = SQLTool()
# mongo_tool= MongoTool()

# Request schema for chat
class ChatRequest(BaseModel):
    user_id: str
    msg: str
    metadata: dict | None = None


@app.get("/")
def home():
    return {"message": "welcome to the Agent"}


@app.post("/chat")
def chat(request: ChatRequest):
    user_id = request.user_id
    message = request.msg
    meta = request.metadata or {}

    if not user_id or not isinstance(user_id, str):
        raise HTTPException(status_code=400, detail="Missing user_id")

    top_k = int(meta.get("top_k", DEFAULT_TOP_K))
    threshold = float(meta.get("relevance_threshold", DEFAULT_RELEVANCE))

    # ---- 1) intent ----
    intent = classify_intent(message)

    # ---- 2) memory ----
    mem = memory.get(user_id)

    tool_steps = []
    rag_context = None
    sql_rows = None

    if intent == "Ecommerce_Info":
        results = retriever.retrieve(message, k=top_k)
        tool_steps.append({"step": "vector_retrieve", "n": len(results)})
        good = [r for r in results if (1.0 - r["score"]) >= threshold or r["score"] >= threshold]
        rag_context = [r["text"] for r in (good or results)]

    elif intent == "Ecommerce_Data":
        out = sql_tool.run(message)
        tool_steps.append({
            "step": "sql_tool",
            "sql": out.get("sql"),
            "rows": len(out.get("rows", [])),
            "note": out.get("note"),
            "error": out.get("error")
        })
        sql_rows = out.get("rows", [])
        if not sql_rows:
            results = retriever.retrieve(message, k=top_k)
            tool_steps.append({"step": "vector_fallback", "n": len(results)})
            good = [r for r in results if (1.0 - r["score"]) >= threshold or r["score"] >= threshold]
            rag_context = [r["text"] for r in (good or results)]

    else:
        results = retriever.retrieve(message, k=top_k)
        tool_steps.append({"step": "vector_default", "n": len(results)})
        rag_context = [r["text"] for r in results]

    # ---- 3) build answer or fallback ----
    has_ctx = (rag_context and any(rag_context)) or (sql_rows and len(sql_rows) > 0)
    if not has_ctx:
        answer = "I couldn't find relevant data for that in my sources (vector DB or Postgres). Could you rephrase or provide more details?"
    else:
        answer = build_answer(message, mem, rag_context, sql_rows)

    # ---- 4) update memory ----
    memory.append(user_id, "user", message)
    memory.append(user_id, "assistant", answer)

    # ---- 5) return JSON ----
    # return {
    #     "session_id": user_id,
    #     "intent": intent,
    #     "answer": answer,
    #     "used_context": {
    #         "rag_count": len(rag_context) if rag_context else 0,
    #         "sql_rows": len(sql_rows) if sql_rows else 0
    #     },
    # }
    return answer


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
