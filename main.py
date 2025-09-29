import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv

from agent.memory import make_checkpointer
from agent.graph import build_graph
from agent.tools import make_tools

load_dotenv()

# Build graph + tools once at startup
_tools = make_tools()
_checkpointer = make_checkpointer()
_graph = build_graph(_tools, _checkpointer)

app = Flask(__name__)

@app.post("/recommend")
def recommend():
    payload = request.get_json(force=True) or {}
    q = payload.get("query", "")
    thread = payload.get("user_id")
    from services.recommender import run_query
    out, tid = run_query(_graph, _checkpointer, q, thread_id=thread)
    out["_thread_id"] = tid
    return jsonify(out)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), debug=True)
