from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from agent.memory import RollingMemory
from agent.tools.vector_pinecone import VectorRetriever
from agent.tools.sql_postgres import SQLTool
# from agent.tools.mongo_tool import MongoTool
# from agent.router.intent import classify_intent
# from agent.router.response_builder import build_answer

from acp.routers.agents import router as acp_agents_router
from acp.routers.runs import router as acp_runs_router

# Load env
load_dotenv()


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


app.include_router(acp_agents_router)
app.include_router(acp_runs_router)


@app.get("/")
def home():
    return {"message": "welcome to the Agent"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
