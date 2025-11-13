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
user_configs_coll = collection["Configuration"]


class FetchConfigResponse(BaseModel):
    """Response model for fetch configuration"""
    success: bool
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@router.get("/detail", response_model=FetchConfigResponse)
async def fetch_configuration(request: Request, user_id: Optional[str] = None):
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
        # Get user_id from query parameter or request headers
        if not user_id:
            # Try to get from Authorization header or request state (if JWT is being used)
            user_id = request.headers.get("X-User-ID") or getattr(request.state, "user_id", None)
        
        if not user_id:
            raise HTTPException(
                status_code=400,
                detail="user_id is required either as query parameter or in request headers"
            )
        
        # Query MongoDB with filters
        query = {
            "user_id": user_id,
            "processed": True,
            "scrape_status": "completed"
        }
        
        # Find configuration matching the criteria
        config = user_configs_coll.find_one(query, {"_id": 0})
        
        if not config:
            return FetchConfigResponse(
                success=False,
                message=f"No completed configuration found for user_id: {user_id}",
                data=None
            )
        
        return FetchConfigResponse(
            success=True,
            message="Configuration retrieved successfully",
            data=config
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching configuration: {str(e)}"
        )


@router.post("/fetch-configuration", response_model=FetchConfigResponse)
async def fetch_configuration_post(request: Request):
    """
    POST version: Fetch configuration from MongoDB based on user_id from request body
    
    Filters:
    - processed: true
    - scrape_status: "completed"
    
    Request body should contain:
    {
        "user_id": "string"
    }
    
    Returns:
        Configuration data matching the criteria
    """
    try:
        # Parse request body
        body = await request.json()
        user_id = body.get("user_id")
        
        if not user_id:
            raise HTTPException(
                status_code=400,
                detail="user_id is required in request body"
            )
        
        # Query MongoDB with filters
        query = {
            "user_id": user_id,
            "processed": True,
            "scrape_status": "completed"
        }
        
        # Find configuration matching the criteria
        config = user_configs_coll.find_one(query, {"_id": 0})
        
        if not config:
            return FetchConfigResponse(
                success=False,
                message=f"No completed configuration found for user_id: {user_id}",
                data=None
            )
        
        return FetchConfigResponse(
            success=True,
            message="Configuration retrieved successfully",
            data=config
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching configuration: {str(e)}"
        )


@router.get("/fetch-all-configurations")
async def fetch_all_configurations(request: Request, user_id: Optional[str] = None):
    """
    Fetch all completed configurations for a user (returns list)
    
    Args:
        request: FastAPI request object
        user_id: Optional user_id as query parameter
        
    Returns:
        List of all configurations matching the criteria
    """
    try:
        # Get user_id from query parameter or request headers
        if not user_id:
            user_id = request.headers.get("X-User-ID") or getattr(request.state, "user_id", None)
        
        if not user_id:
            raise HTTPException(
                status_code=400,
                detail="user_id is required either as query parameter or in request headers"
            )
        
        # Query MongoDB with filters
        query = {
            "user_id": user_id,
            "processed": True,
            "scrape_status": "completed"
        }
        
        # Find all configurations matching the criteria
        configs = list(user_configs_coll.find(query, {"_id": 0}))
        
        return {
            "success": True,
            "message": f"Found {len(configs)} completed configuration(s)",
            "count": len(configs),
            "data": configs
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching configurations: {str(e)}"
        )
