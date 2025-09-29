import os
import json
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings

from config.mongo_con import get_collection
from config.pinecone_con import get_index, get_namespace

DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "6"))

# Reuse OpenAI for query embeddings
_embed = OpenAIEmbeddings()

class FilterSchema(BaseModel):
    query: Optional[str] = Field(None, description="Free text like 'cheap gaming earbuds under 1500'")
    max_price: Optional[float] = Field(None, description="Upper price bound (INR)")
    min_price: Optional[float] = Field(None, description="Lower price bound (INR)")
    category: Optional[str] = Field(None, description="Category, e.g., 'earbuds', 'smartwatch'")
    brand: Optional[str] = Field(None, description="Brand to prefer or require")
    top_k: int = Field(DEFAULT_TOP_K, description="How many candidates to return")

def _shape_items(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for d in docs:
        out.append({
            "id": d.get("id") or str(d.get("_id")),
            "title": d.get("title", ""),
            "brand": d.get("brand", ""),
            "category": d.get("category", ""),
            "price": float(d.get("price", 0)),
            "currency": d.get("currency", "INR"),
            "tags": ", ".join(d.get("tags", [])) if isinstance(d.get("tags"), list) else d.get("tags"),
            "description": d.get("description", ""),
            "score": d.get("_score"),  # optional
        })
    return out

def _mongo_fetch_by_ids(ids: List[str]) -> Dict[str, Dict[str, Any]]:
    col = get_collection()
    docs = list(col.find({"$or": [{"id": {"$in": ids}}, {"_id": {"$in": ids}}]}))
    # Normalize to map by id (prefer 'id' field, fallback to _id)
    byid = {}
    for d in docs:
        key = d.get("id") or str(d.get("_id"))
        byid[key] = d
    return byid

def _apply_filters(doc: Dict[str, Any], max_price, min_price, category, brand) -> bool:
    if max_price is not None and float(doc.get("price", 0)) > max_price:
        return False
    if min_price is not None and float(doc.get("price", 0)) < min_price:
        return False
    if category and str(doc.get("category", "")).lower() != str(category).lower():
        return False
    if brand and str(doc.get("brand", "")).lower() != str(brand).lower():
        return False
    return True

def _json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)

def make_tools():
    @tool("semantic_search_pinecone", args_schema=FilterSchema)
    def semantic_search_pinecone(
        query: Optional[str]=None,
        max_price: Optional[float]=None,
        min_price: Optional[float]=None,
        category: Optional[str]=None,
        brand: Optional[str]=None,
        top_k: int = DEFAULT_TOP_K
    ):
        """
        Semantic search via Pinecone (vectors) + hydrate from MongoDB.
        We:
          1) Embed the query
          2) Query Pinecone for candidate IDs (metadata should include product 'id' if possible)
          3) Fetch fresh documents from MongoDB by IDs
          4) Apply final structured filters
          5) Return compact candidates for the LLM to re-rank
        """
        text = query or "top products"
        qvec = _embed.embed_query(text)

        index = get_index()
        namespace = get_namespace()

        # Pull a wider net; we'll prune later
        pine_res = index.query(
            namespace=namespace,
            vector=qvec,
            top_k=max(top_k * 3, 30),
            include_values=False,
            include_metadata=True,
        )

        # Extract candidate IDs (prefer metadata["id"], fallback to the Pinecone id)
        ids_ordered: List[str] = []
        id_to_score: Dict[str, float] = {}
        for match in pine_res.matches or []:
            pid = (match.metadata.get("id") if match.metadata else None) or match.id
            ids_ordered.append(pid)
            id_to_score[pid] = float(match.score) if match.score is not None else None

        # Hydrate from Mongo
        byid = _mongo_fetch_by_ids(ids_ordered)
        hydrated: List[Dict[str, Any]] = []
        for pid in ids_ordered:
            d = byid.get(pid)
            if not d:
                continue
            d = dict(d)  # copy
            d["_score"] = id_to_score.get(pid)
            if _apply_filters(d, max_price, min_price, category, brand):
                hydrated.append(d)
            if len(hydrated) >= top_k:
                break

        return _json({"candidates": _shape_items(hydrated)})

    @tool("fetch_item_by_id_mongo")
    def fetch_item_by_id_mongo(id: str):
        """
        Fetch full details for a product by ID from MongoDB.
        """
        col = get_collection()
        doc = col.find_one({"$or": [{"id": id}, {"_id": id}]})
        if not doc:
            return _json({"error": "not_found"})
        return _json({
            "id": doc.get("id") or str(doc.get("_id")),
            "title": doc.get("title", ""),
            "brand": doc.get("brand", ""),
            "category": doc.get("category", ""),
            "price": float(doc.get("price", 0)),
            "currency": doc.get("currency", "INR"),
            "tags": ", ".join(doc.get("tags", [])) if isinstance(doc.get("tags"), list) else doc.get("tags"),
            "description": doc.get("description", "")
        })

    return [semantic_search_pinecone, fetch_item_by_id_mongo]
