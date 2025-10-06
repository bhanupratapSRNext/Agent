import os
from typing import List, Tuple, Dict, Any

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

def build_answer(
    message: str,
    memory: List[Tuple[str, str]],
    rag_snippets: List[str] | None,
    sql_rows: List[Dict[str, Any]] | None
) -> str:
    openai_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL")
    if not openai_key:
        # Safe fallback if no LLM configured
        base = "Based on what I have, "
        if sql_rows:
            return base + f"here are some results I found: {sql_rows[:5]}"
        if rag_snippets:
            return base + "I found some related context, but no exact match."
        return "I couldn't find relevant data for that."

    llm = ChatOpenAI(api_key=openai_key, model=model, temperature=0.2)
    sys = (
        "You are a helpful assistant.\n"
        "If specific factual data is missing, say so clearly (e.g., 'I couldn't find relevant data for that').\n"
        "Use the provided context faithfully; do not fabricate."
    )
    parts = []
    if memory:
        parts.append("Previous conversation:\n" + "\n".join([f"{r}: {t}" for r,t in memory]))
    if rag_snippets:
        parts.append("Vector DB context:\n" + "\n---\n".join(rag_snippets))
    if sql_rows:
        parts.append("SQL results (tabular):\n" + str(sql_rows[:5]))
    ctx = "\n\n".join(parts) if parts else "(no context)"

    return llm.invoke([
        {"role": "system", "content": sys},
        {"role": "user", "content": f"User: {message}\n\nContext:\n{ctx}"}
    ]).content.strip()
