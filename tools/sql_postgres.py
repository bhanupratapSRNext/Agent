import os
from typing import Dict, Any

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine, text
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain

def _pg_uri() -> str:
    host = os.getenv("PG_HOST")
    port = int(os.getenv("PG_PORT"))
    db   = os.getenv("PG_DB")
    user = os.getenv("PG_USER")
    pwd  = os.getenv("PG_PASSWORD")
    return f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"

class SQLTool:
    def __init__(self):
        # Try OpenRouter first, fallback to OpenAI
        api_key = os.getenv("OPEN_ROUTER_KEY")
        if not api_key:
            raise RuntimeError("Missing OPEN_ROUTER_KEY environment variable")

        # Get model from environment or use default
        llm_model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        
        self.engine = create_engine(_pg_uri(), pool_pre_ping=True)
        self.db = SQLDatabase.from_uri(_pg_uri())
        
        # Use OpenRouter if OPEN_ROUTER_KEY is set
        if os.getenv("OPEN_ROUTER_KEY"):
            self.llm = ChatOpenAI(
                api_key=api_key,
                model=llm_model,
                temperature=0,
                base_url="https://openrouter.ai/api/v1",
                model_kwargs={
                    "response_format": {"type": "text"}
                }
            )
        else:
            # Fallback to direct OpenAI
            self.llm = ChatOpenAI(api_key=api_key, model=llm_model, temperature=0)

        allowlist = os.getenv("SQL_ALLOWED_TABLES", "")
        self.allowed_tables = [t.strip().lower() for t in allowlist.split(",") if t.strip()]

    def run(self, question: str) -> Dict[str, Any]:
        chain = create_sql_query_chain(self.llm, self.db)
        sql = chain.invoke({"question": question}).strip()
        
        # Simple cleanup for OpenRouter responses
        # Remove common prefixes and markdown code blocks
        if sql.startswith("SQLQuery:") or sql.startswith("SQL Query:"):
            sql = sql.split(":", 1)[1].strip()
        
        # Remove markdown code blocks
        sql = sql.replace("```sql", "").replace("```", "").strip()
        
        # If multi-line, join lines
        if "\n" in sql:
            sql = " ".join(line.strip() for line in sql.split("\n") if line.strip())

        if self.allowed_tables:
            lowered = sql.lower()
            for t in self.allowed_tables:
                if f" {t} " in lowered or f" {t}(" in lowered or f" {t}\n" in lowered:
                    break
            else:
                return {"sql": sql, "rows": [], "note": "Blocked by table allowlist"}

        with self.engine.connect() as conn:
            try:
                result = conn.execute(text(sql))
                rows = [dict(r._mapping) for r in result]
                return {"sql": sql, "rows": rows}
            except Exception as e:
                return {"sql": sql, "rows": [], "error": str(e)}
