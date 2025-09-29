# pip install pymongo
import os
from pymongo import MongoClient

MONGO_URI = os.getenv("MONGO_URI")  

client = MongoClient(
    MONGO_URI,
    maxPoolSize=50,          
    minPoolSize=5,           
    waitQueueTimeoutMS=5000, 
    serverSelectionTimeoutMS=3000,
    connectTimeoutMS=3000,
    socketTimeoutMS=10000,
)



from functools import lru_cache
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB", "products_db")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "catalog")
MONGO_VECTOR_INDEX_NAME = os.getenv("MONGO_VECTOR_INDEX_NAME", "product_idx")

@lru_cache(maxsize=1)
def get_client() -> MongoClient:
    return MongoClient(MONGODB_URI)

def get_collection():
    client = get_client()
    return client[MONGODB_DB][MONGODB_COLLECTION]

def vector_index_name() -> str:
    return MONGO_VECTOR_INDEX_NAME
