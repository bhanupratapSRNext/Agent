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
        openai_key = os.getenv("OPENAI_API_KEY")
        llm_model = os.getenv("OPENAI_MODEL")
        if not openai_key:
            raise RuntimeError("Missing OPENAI_API_KEY")

        self.engine = create_engine(_pg_uri(), pool_pre_ping=True)
        self.db = SQLDatabase.from_uri(_pg_uri())
        self.llm = ChatOpenAI(api_key=openai_key, model=llm_model, temperature=0)

        allowlist = os.getenv("SQL_ALLOWED_TABLES", "")
        self.allowed_tables = [t.strip().lower() for t in allowlist.split(",") if t.strip()]

    def run(self, question: str) -> Dict[str, Any]:
        chain = create_sql_query_chain(self.llm, self.db)
        sql = chain.invoke({"question": question}).strip()

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
