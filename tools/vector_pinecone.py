# import os
# from typing import List, Dict, Any

# from dotenv import load_dotenv
# load_dotenv()

# from langchain_openai import OpenAIEmbeddings
# from langchain_pinecone import PineconeVectorStore
# from pinecone import Pinecone
# from langchain_community.embeddings import HuggingFaceEmbeddings

# class VectorRetriever:
#     def __init__(self):
#         api_key = os.getenv("OPEN_ROUTER_KEY")
#         embed_model = os.getenv("EMBED_MODEL")
#         pinecone_key = os.getenv("PINECONE_API_KEY")
#         index_name = os.getenv("PINECONE_INDEX_NAME")
#         namespace = os.getenv("PINECONE_NAMESPACE", "default")

#         if not (api_key and pinecone_key and index_name):
#             raise RuntimeError("Missing OPEN_ROUTER_KEY or PINECONE_* env vars")

#         self.pc = Pinecone(api_key=pinecone_key)
#         # self.embeddings = OpenAIEmbeddings(api_key=openai_key, model=embed_model)
#         self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#         self.vs = PineconeVectorStore.from_existing_index(
#             index_name=index_name,
#             namespace=namespace,
#             embedding=self.embeddings,
#         )

#     def retrieve(self, query: str, k: int) -> List[Dict[str, Any]]:
#         docs = self.vs.similarity_search_with_score(query, k=k)
#         return [{"text": d.page_content, "metadata": d.metadata, "score": float(s)} for d, s in docs]
