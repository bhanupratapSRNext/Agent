from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
import os
from typing import Optional
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import tempfile
import shutil
from pathlib import Path
from config.mongo_con import client

try:
    from pinecone.grpc import PineconeGRPC as Pinecone
except ImportError:
    from pinecone import Pinecone
    
from pinecone import ServerlessSpec

load_dotenv()

router = APIRouter(prefix="/configure", tags=["Pinecone Management"])


class CreateIndexRequest(BaseModel):
    """Request model for creating a Pinecone index"""
    index_name: str = Field(..., description="Name of the index to create")
    dimension: int = Field(..., description="Dimension of the vectors", ge=1, le=20000)
    metric: str = Field(default="cosine", description="Distance metric (cosine, euclidean, dotproduct)")
    cloud: str = Field(default="aws", description="Cloud provider (aws, gcp, azure)")
    region: str = Field(default="us-east-1", description="Cloud region")
    api_key: Optional[str] = Field(None, description="Optional API key (uses env var if not provided)")
    user_id: str = Field(..., description="User ID associated with this request")


class CreateIndexResponse(BaseModel):
    """Response model for index creation"""
    success: bool
    message: str
    index_name: str
    details: Optional[dict] = None


class IndexInfo(BaseModel):
    """Model for index information"""
    name: str
    dimension: int
    metric: str
    host: str
    status: str


class ListIndexesResponse(BaseModel):
    """Response model for listing indexes"""
    success: bool
    indexes: list[IndexInfo]
    count: int


class DeleteIndexRequest(BaseModel):
    """Request model for deleting an index"""
    name: str = Field(..., description="Name of the index to delete")
    api_key: Optional[str] = Field(None, description="Optional API key (uses env var if not provided)")


class DeleteIndexResponse(BaseModel):
    """Response model for index deletion"""
    success: bool
    message: str
    index_name: str


class CreateEmbeddingsRequest(BaseModel):
    """Request model for creating embeddings from PDF files and pushing to Pinecone"""
    data_dir: str = Field("Data", description="Directory path containing PDF files (glob *.pdf)")
    index_name: str = Field(..., description="Pinecone index name to write embeddings into")
    chunk_size: int = Field(500, ge=64, le=2000, description="Chunk size for text splitting")
    chunk_overlap: int = Field(20, ge=0, le=500, description="Chunk overlap for text splitting")
    model_name: str = Field("sentence-transformers/all-MiniLM-L6-v2", description="HuggingFace embedding model name")
    api_key: Optional[str] = Field(None, description="Optional Pinecone API key (uses env var if not provided)")


class CreateEmbeddingsResponse(BaseModel):
    success: bool
    message: str
    index_name: str
    upserted_count: int | None = None


@router.post("/create-index", response_model=CreateIndexResponse, status_code=status.HTTP_201_CREATED)
async def create_index(request: CreateIndexRequest):
    try:
        # Use provided API key or fall back to environment variable
        api_key = request.api_key or os.getenv("PINECONE_API_KEY")
        
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Pinecone API key not provided and not found in environment variables"
            )
        
        # Initialize Pinecone client
        pc = Pinecone(api_key=api_key)
        
        # List all indexes and check if index already exists (same logic as /list-indexes)
        indexes_list = pc.list_indexes()
        existing_indexes = [idx["name"] for idx in indexes_list.indexes]
        
        # Check if index already exists in Pinecone
        if request.index_name in existing_indexes:
            return CreateIndexResponse(
                success=False,
                message=f"Index '{request.index_name}' already exists in Pinecone",
                index_name=request.index_name,
                details={
                    "dimension": request.dimension,
                    "metric": request.metric,
                    "cloud": request.cloud,
                    "region": request.region,
                    "already_exists": True,
                }
            )
        
        # Validate metric
        valid_metrics = ["cosine", "euclidean", "dotproduct"]
        if request.metric not in valid_metrics:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid metric. Must be one of: {', '.join(valid_metrics)}"
            )
        
        # Import MongoDB client
        
        
        # MongoDB setup
        collection = client["chat-bot"]
        pinecone_indexes_coll = collection["Configuration"]
        
        # Save index metadata to MongoDB before creating in Pinecone
        index_metadata = {
            "index_name": request.index_name,
            "dimension": request.dimension,
            "metric": request.metric,
            "cloud": request.cloud,
            "region": request.region,
            "created_at": None,  # Will be updated after successful creation
            "status": "creating",
            "user_id":request.user_id,
            "processed":False
        }
        
        # Insert into MongoDB
        mongo_result = pinecone_indexes_coll.insert_one(index_metadata)
        
        return CreateIndexResponse(
            success=True,
            message=f"Index '{request.index_name}' successfully saved to MongoDB",
            index_name=request.index_name,
            details={
                "dimension": request.dimension,
                "metric": request.metric,
                "cloud": request.cloud,
                "region": request.region,
                "mongo_id": str(mongo_result.inserted_id)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating index: {str(e)}"
        )




class SaveURLRequest(BaseModel):
    """Request model for saving URL to Configuration"""
    root_url: str = Field(..., description="Website URL to save")
    user_id: Optional[str] = Field(None, description="Optional user ID")


class SaveURLResponse(BaseModel):
    """Response model for URL save operation"""
    success: bool
    message: str
    url: str
    mongo_id: Optional[str] = None


@router.post("/save", response_model=SaveURLResponse)
async def save_url(request: SaveURLRequest):
    """
    Save or update URL in MongoDB Configuration collection based on user_id
    
    Args:
        request: SaveURLRequest with URL and user_id
        
    Returns:
        SaveURLResponse with save/update status
    """
    try:
        # MongoDB setup
        collection = client["chat-bot"]
        configuration_coll = collection["Configuration"]
        
        # Validate that user_id is provided
        if not request.user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="user_id is required to save URL"
            )
        
        # Check if document exists for this user_id
        existing_config = configuration_coll.find_one({"user_id": request.user_id})
        
        from datetime import datetime
        
        if existing_config:
            # Update existing document
            update_data = {
                "root_url": request.root_url,
                "updated_at": datetime.utcnow(),
                "status": "updated"
            }
            
            configuration_coll.update_one(
                {"user_id": request.user_id},
                {"$set": update_data}
            )
            
            return SaveURLResponse(
                success=True,
                message=f"URL '{request.root_url}' updated successfully for user_id '{request.user_id}'",
                url=request.root_url,
                mongo_id=str(existing_config["_id"])
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving URL: {str(e)}"
        )




















# @router.get("/list-indexes", response_model=ListIndexesResponse)
# async def list_indexes(api_key: Optional[str] = None):
#     """
#     List all Pinecone indexes
    
#     Args:
#         api_key: Optional API key (uses env var if not provided)
        
#     Returns:
#         ListIndexesResponse with list of indexes
        
#     Raises:
#         HTTPException: If listing fails
#     """
#     try:
#         # Use provided API key or fall back to environment variable
#         api_key = api_key or os.getenv("PINECONE_API_KEY")
        
#         if not api_key:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Pinecone API key not provided and not found in environment variables"
#             )
        
#         # Initialize Pinecone client
#         pc = Pinecone(api_key=api_key)
        
#         # List all indexes
#         indexes_list = pc.list_indexes()
        
#         # Format the response
#         indexes = [
#             IndexInfo(
#                 name=idx["name"],
#                 dimension=idx["dimension"],
#                 metric=idx["metric"],
#                 host=idx["host"],
#                 status=idx.get("status", {}).get("ready", "unknown")
#             )
#             for idx in indexes_list.indexes
#         ]
        
#         return ListIndexesResponse(
#             success=True,
#             indexes=indexes,
#             count=len(indexes)
#         )
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error listing indexes: {str(e)}"
#         )


# @router.delete("/delete-index", response_model=DeleteIndexResponse)
# async def delete_index(request: DeleteIndexRequest):
#     """
#     Delete a Pinecone index
    
#     Args:
#         request: DeleteIndexRequest with index name
        
#     Returns:
#         DeleteIndexResponse with deletion status
        
#     Raises:
#         HTTPException: If deletion fails
#     """
#     try:
#         # Use provided API key or fall back to environment variable
#         api_key = request.api_key or os.getenv("PINECONE_API_KEY")
        
#         if not api_key:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Pinecone API key not provided and not found in environment variables"
#             )
        
#         # Initialize Pinecone client
#         pc = Pinecone(api_key=api_key)
        
#         # Check if index exists
#         existing_indexes = [idx["name"] for idx in pc.list_indexes().indexes]
#         if request.name not in existing_indexes:
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail=f"Index '{request.name}' not found"
#             )
        
#         # Delete the index
#         pc.delete_index(request.name)
        
#         return DeleteIndexResponse(
#             success=True,
#             message=f"Index '{request.name}' deleted successfully",
#             index_name=request.name
#         )
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error deleting index: {str(e)}"
#         )


# @router.get("/index-stats/{index_name}")
# async def get_index_stats(index_name: str, api_key: Optional[str] = None):
#     """
#     Get statistics for a specific Pinecone index
    
#     Args:
#         index_name: Name of the index
#         api_key: Optional API key (uses env var if not provided)
        
#     Returns:
#         dict with index statistics
        
#     Raises:
#         HTTPException: If stats retrieval fails
#     """
#     try:
#         # Use provided API key or fall back to environment variable
#         api_key = api_key or os.getenv("PINECONE_API_KEY")
        
#         if not api_key:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Pinecone API key not provided and not found in environment variables"
#             )
        
#         # Initialize Pinecone client
#         pc = Pinecone(api_key=api_key)
        
#         # Check if index exists
#         existing_indexes = [idx["name"] for idx in pc.list_indexes().indexes]
#         if index_name not in existing_indexes:
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail=f"Index '{index_name}' not found"
#             )
        
#         # Get index and its stats
#         index = pc.Index(index_name)
#         stats = index.describe_index_stats()
        
#         return {
#             "success": True,
#             "index_name": index_name,
#             "stats": stats
#         }
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error getting index stats: {str(e)}"
#         )



class UploadPDFResponse(BaseModel):
    """Response model for PDF upload and embedding creation"""
    success: bool
    message: str
    index_name: str
    filename: str
    chunks_created: int
    upserted_count: int
    
@router.post("/create-embeddings", response_model=UploadPDFResponse)
async def test():
    return {"success": True, "message": "Test endpoint working", "index_name": "test_index", "filename": "test.pdf", "chunks_created": 0, "upserted_count": 0}

# @router.post("/create-embeddings", response_model=UploadPDFResponse)
# async def upload_pdf_and_create_embeddings(
#     file: UploadFile = File(..., description="PDF file to process"),
#     index_name: str = Form(..., description="Pinecone index name to store embeddings"),
#     chunk_size: int = Form(500, description="Chunk size for text splitting"),
#     chunk_overlap: int = Form(20, description="Chunk overlap for text splitting"),
#     model_name: str = Form("sentence-transformers/all-MiniLM-L6-v2", description="HuggingFace embedding model"),
#     api_key: Optional[str] = Form(None, description="Optional Pinecone API key")
# ):
#     temp_dir = None
#     try:
#         # Validate file type
#         if not file.filename.lower().endswith('.pdf'):
#             return "Only PDF files are supported"
        
#         # Resolve API key
#         resolved_api_key = api_key or os.getenv("PINECONE_API_KEY")
#         if not resolved_api_key:
#             return "PINECONE_API_KEY not provided"
        
#         # Initialize Pinecone client and verify index exists
#         pc = Pinecone(api_key=resolved_api_key)
#         existing_indexes = [idx["name"] for idx in pc.list_indexes().indexes]
#         if index_name not in existing_indexes:
#             return f"Index '{index_name}' not found. Create it first via /pinecone/create-index"
        
#         # Create temporary directory for PDF processing
#         temp_dir = tempfile.mkdtemp()
#         temp_file_path = Path(temp_dir) / file.filename
        
#         # Save uploaded file to temporary location
#         with open(temp_file_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
        
#         # Load PDF using PyPDFLoader
#         loader = PyPDFLoader(str(temp_file_path))
#         documents = loader.load()
        
#         if not documents:
#             return "No content could be extracted from the PDF"
        
#         # Split documents into chunks
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=chunk_size,
#             chunk_overlap=chunk_overlap
#         )
#         chunks = text_splitter.split_documents(documents)
        
#         if not chunks:
#             return "No text chunks created from the PDF"
            
#         # Create embeddings
#         embeddings = HuggingFaceEmbeddings(model_name=model_name)
        
#         # Store in Pinecone using LangChain helper
#         docsearch = PineconeVectorStore.from_documents(
#             documents=chunks,
#             index_name=index_name,
#             embedding=embeddings,
#         )
        
#         return UploadPDFResponse(
#             success=True,
#             message=f"Successfully processed '{file.filename}' and stored embeddings in Pinecone",
#             index_name=index_name,
#             filename=file.filename,
#             chunks_created=len(chunks),
#             upserted_count=len(chunks)
#         )
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         import traceback
#         error_trace = traceback.format_exc()
#         print(f"Error processing PDF: {error_trace}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error processing PDF: {str(e)}"
#         )
#     finally:
#         if temp_dir and Path(temp_dir).exists():
#             shutil.rmtree(temp_dir, ignore_errors=True)



