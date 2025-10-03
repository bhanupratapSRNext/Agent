from typing import Dict
from .models import AgentManifest

# public registry other systems discover
AGENTS: Dict[str, AgentManifest] = {
    "router": AgentManifest(
        name="router",
        description="Routes e-commerce queries to info (RAG) or data (SQL) agents; composes final answer.",
        metadata={"capabilities": ["routing","composition"], "natural_languages":["en"]}
    ),
    "ecommerce-info": AgentManifest(
        name="ecommerce-info",
        description="Answers product/category/FAQ style questions via vector RAG (Pinecone).",
        metadata={"capabilities": ["RAG","Pinecone"]}
    ),
    "ecommerce-data": AgentManifest(
        name="ecommerce-data",
        description="Answers account/order/payment queries via Postgres SQLTool with RAG fallback.",
        metadata={"capabilities": ["Postgres","SQL","RAG-fallback"]}
    ),
}
