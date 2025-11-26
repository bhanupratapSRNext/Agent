"""
Fetch Configuration API
API endpoint to fetch completed and processed configurations from MongoDB
"""
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# Import MongoDB client
from config.mongo_con import client

load_dotenv()

# Create FastAPI router
router = APIRouter(prefix="/fetch-configuration", tags=["configuration"])
# MongoDB setup
collection = client["chat-bot"]
script_configs = collection["script"]


class FetchConfigResponse(BaseModel):
    """Response model for fetch configuration"""
    success: bool
    user_id: str
    message: Optional[str] = None
    data: str = None
    error: Optional[str] = None

class FetchConfigRequest(BaseModel):
    """Request model for fetch configuration"""
    user_id: str


class FetchConfigRequest(BaseModel):
    """Request model for fetch configuration"""
    user_id: str


class ConfigItem(BaseModel):
    """Single configuration item summary"""
    Index: str
    Status: Optional[str] = None
    URL: Optional[str] = None
    scrape_status: Optional[bool] = False
    # you can add more fields here if needed


class FetchConfigListResponse(BaseModel):
    """Response model for listing configurations"""
    success: bool
    user_id: str
    message: Optional[str] = None
    data: List[ConfigItem] = []
    error: Optional[str] = None


@router.post("/detail", response_model=FetchConfigResponse)
async def fetch_configuration(request: FetchConfigRequest):
    """
    Fetch configuration from MongoDB based on user_id with filters:
    - processed: true
    - scrape_status: "completed"
    
    Args:
        request: FastAPI request object (to get user_id from headers/auth if needed)
        user_id: Optional user_id as query parameter
        
    Returns:
        Configuration data matching the criteria
    """
    try:   
        user_id = request.user_id
        
        if not user_id:
            raise HTTPException(
                status_code=400,
                detail="user_id is required either as query parameter or in request headers"
            )
        
        config_doc = script_configs.find_one({"user_id": user_id})
        
        if not config_doc:
            return FetchConfigResponse(
                success=False,
                message=f"No completed configuration found for user_id: {user_id}",
                data=None
            )
         # Convert ObjectId to string
        script_text = config_doc.get("script", None)

        return FetchConfigResponse(
            success=True,
            user_id=user_id,
            message="Configuration fetched successfully",
            data=script_text
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching configuration: {str(e)}"
        )




@router.post("/list", response_model=FetchConfigListResponse)
async def list_configurations(request: FetchConfigRequest):
    """
    List all configurations for a given user_id.
    You can optionally filter by processed/scrape_status.
    """
    try:
        user_id = request.user_id

        if not user_id:
            raise HTTPException(
                status_code=400,
                detail="user_id is required"
            )

        query = {"user_id": user_id}
        Configuration = collection["Configuration"]
        cursor = Configuration.find(query)

        configs: List[ConfigItem] = []

        for doc in cursor:
            configs.append(
                ConfigItem(
                    Index=str(doc.get("index_name")),
                    Status=doc.get("progress"),
                    URL=doc.get("root_url")
                )
            )

        if not configs:
            return FetchConfigListResponse(
                success=False,
                user_id=user_id,
                message=f"No configurations found for user_id: {user_id}",
                data=[]
            )

        return FetchConfigListResponse(
            success=True,
            user_id=user_id,
            message=f"Found {len(configs)} configurations for user_id: {user_id}",
            data=configs
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing configurations: {str(e)}"
        )