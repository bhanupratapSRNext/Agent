import os
from functools import lru_cache
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE") or None  # None uses root ns

@lru_cache(maxsize=1)
def _pc() -> Pinecone:
    return Pinecone(api_key=PINECONE_API_KEY)

def get_index():
    pc = _pc()
    # Expect index to exist. If you want auto-create, uncomment:
    # if PINECONE_INDEX_NAME not in [i["name"] for i in pc.list_indexes().indexes]:
    #     pc.create_index(
    #         name=PINECONE_INDEX_NAME,
    #         dimension=1536,  # match your embedding model
    #         metric="cosine",
    #         spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    #     )
    return pc.Index(PINECONE_INDEX_NAME)

def get_namespace():
    return PINECONE_NAMESPACE


# def create_pinecone_index():
#     try:
#         pc.create_index(
#             name=index_name,
#             dimension=384, 
#             metric="cosine", 
#             spec=ServerlessSpec(
#                 cloud="aws", 
#                 region="us-east-1"
#             ) 
#         )
#         print(f"Index '{index_name}' created successfully")
#     except Exception as e:
#         print(f"Error creating index: {str(e)}") 