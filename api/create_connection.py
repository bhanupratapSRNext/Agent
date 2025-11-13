# """
# Database Connection Management API
# Handles storage and retrieval of database credentials for SQL and NoSQL databases
# """
# import os
# from typing import Optional, Literal, Union
# from datetime import datetime
# from fastapi import APIRouter, HTTPException, status
# from pydantic import BaseModel, Field, field_validator, model_validator
# from bson import ObjectId
# from dotenv import load_dotenv
# from pymongo import MongoClient

# load_dotenv()

# router = APIRouter(prefix="/connections", tags=["Database Connections"])

# # MongoDB configuration
# MONGODB_URI = os.getenv("MONGODB_URI") or os.getenv("MONGO_URI")
# MONGODB_DB = os.getenv("MONGODB_DB", "configuration")
# CONNECTIONS_COLLECTION = "db_connections"


# def get_mongo_collection():
#     """Get MongoDB collection for storing connection credentials"""
#     if not MONGODB_URI:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="MongoDB URI not configured. Set MONGODB_URI or MONGO_URI in environment variables"
#         )
#     client = MongoClient(MONGODB_URI)
#     db = client[MONGODB_DB]
#     return db[CONNECTIONS_COLLECTION]


# # Pydantic Models
# class SQLCredentials(BaseModel):
#     """SQL Database connection credentials"""
#     db_type: Literal["postgresql", "mysql", "sqlite", "mssql", "oracle"] = Field(..., description="Type of SQL database")
#     host: str = Field(..., description="Database host/server address")
#     port: int = Field(..., description="Database port", ge=1, le=65535)
#     username: str = Field(..., description="Database username")
#     password: str = Field(..., description="Database password")
#     database: str = Field(..., description="Database name")
#     additional_params: Optional[dict] = Field(None, description="Additional connection parameters")
    
#     @field_validator('db_type')
#     @classmethod
#     def validate_db_type(cls, v):
#         valid_types = ["postgresql", "mysql", "sqlite", "mssql", "oracle"]
#         if v not in valid_types:
#             raise ValueError(f"db_type must be one of {valid_types}")
#         return v


# class NoSQLCredentials(BaseModel):
#     """NoSQL Database connection credentials"""
#     connection_url: str = Field(..., description="MongoDB connection URL/URI")


# class SaveConnectionRequest(BaseModel):
#     """Request model for saving database connection"""
#     connection_name: str = Field(..., description="Unique name for this connection", min_length=1)
#     connection_type: Literal["sql", "nosql"] = Field(..., description="Type of database connection")
#     sql_credentials: Optional[SQLCredentials] = Field(None, description="SQL database credentials")
#     nosql_credentials: Optional[NoSQLCredentials] = Field(None, description="NoSQL database credentials")
#     description: Optional[str] = Field(None, description="Description of this connection")
#     is_active: bool = Field(True, description="Whether this connection is active")
    
#     @model_validator(mode='after')
#     def validate_credentials(self):
#         """Ensure the correct credentials are provided based on connection_type"""
#         if self.connection_type == 'sql':
#             if self.sql_credentials is None:
#                 raise ValueError("sql_credentials required when connection_type is 'sql'")
#             if self.nosql_credentials is not None:
#                 raise ValueError("nosql_credentials should not be provided when connection_type is 'sql'")
        
#         elif self.connection_type == 'nosql':
#             if self.nosql_credentials is None:
#                 raise ValueError("nosql_credentials required when connection_type is 'nosql'")
#             if self.sql_credentials is not None:
#                 raise ValueError("sql_credentials should not be provided when connection_type is 'nosql'")
        
#         return self


# class SaveConnectionResponse(BaseModel):
#     """Response model for saved connection"""
#     success: bool
#     message: str
#     connection_id: str
#     connection_name: str


# class ConnectionInfo(BaseModel):
#     """Model for connection information (without sensitive data by default)"""
#     connection_id: str
#     connection_name: str
#     connection_type: str
#     db_type: str
#     description: Optional[str]
#     is_active: bool
#     created_at: str
#     updated_at: str


# class ListConnectionsResponse(BaseModel):
#     """Response model for listing connections"""
#     success: bool
#     count: int
#     connections: list[ConnectionInfo]


# class DeleteConnectionResponse(BaseModel):
#     """Response model for connection deletion"""
#     success: bool
#     message: str
#     connection_id: str


# # API Endpoints
# @router.post("/save", response_model=SaveConnectionResponse, status_code=status.HTTP_201_CREATED)
# async def save_connection(request: SaveConnectionRequest):
#     try:
#         collection = get_mongo_collection()
        
#         # Check if index name already exists
#         existing = collection.find_one({"Index_name": request.connection_name})
#         if existing:
#             raise HTTPException(
#                 status_code=status.HTTP_409_CONFLICT,
#                 detail=f"Connection with name '{request.connection_name}' already exists"
#             )
        
#         # Prepare document for MongoDB
#         connection_doc = {
#             "Index_name": request.connection_name,
#             "connection_type": request.connection_type,
#             "description": request.description,
#             "is_active": request.is_active,
#             "created_at": datetime.utcnow(),
#             "updated_at": datetime.utcnow()
#         }
        
#         # Add credentials based on type
#         if request.connection_type == "sql":
#             connection_doc["sql_credentials"] = request.sql_credentials.dict()
#             db_type = request.sql_credentials.db_type
#         else:  # nosql
#             connection_doc["nosql_credentials"] = request.nosql_credentials.dict()
#             db_type = "mongodb"  # Since we only support MongoDB for NoSQL
        
#         connection_doc["db_type"] = db_type
        
#         # Insert into MongoDB
#         result = collection.insert_one(connection_doc)
        
#         return SaveConnectionResponse(
#             success=True,
#             message=f"Connection '{request.connection_name}' saved successfully",
#             connection_id=str(result.inserted_id),
#             connection_name=request.connection_name
#         )
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error saving connection: {str(e)}"
#         )


# @router.get("/list", response_model=ListConnectionsResponse)
# async def list_connections(
#     connection_type: Optional[Literal["sql", "nosql"]] = None,
#     is_active: Optional[bool] = None
# ):
#     """
#     List all saved database connections
    
#     Args:
#         connection_type: Optional filter by connection type (sql or nosql)
#         is_active: Optional filter by active status
        
#     Returns:
#         ListConnectionsResponse with list of connections
#     """
#     try:
#         collection = get_mongo_collection()
        
#         # Build query filter
#         query = {}
#         if connection_type:
#             query["connection_type"] = connection_type
#         if is_active is not None:
#             query["is_active"] = is_active
        
#         # Fetch connections (excluding sensitive password data)
#         connections = list(collection.find(query))
        
#         # Format response
#         connection_list = []
#         for conn in connections:
#             connection_list.append(ConnectionInfo(
#                 connection_id=str(conn["_id"]),
#                 connection_name=conn["connection_name"],
#                 connection_type=conn["connection_type"],
#                 db_type=conn["db_type"],
#                 description=conn.get("description"),
#                 is_active=conn["is_active"],
#                 created_at=conn["created_at"].isoformat(),
#                 updated_at=conn["updated_at"].isoformat()
#             ))
        
#         return ListConnectionsResponse(
#             success=True,
#             count=len(connection_list),
#             connections=connection_list
#         )
        
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error listing connections: {str(e)}"
#         )


# @router.get("/{connection_id}")
# async def get_connection(connection_id: str, include_credentials: bool = False):
#     """
#     Get a specific database connection by ID
    
#     Args:
#         connection_id: MongoDB ObjectId of the connection
#         include_credentials: Whether to include sensitive credentials in response (default: False)
        
#     Returns:
#         Connection details
        
#     Raises:
#         HTTPException: If connection not found
#     """
#     try:
#         collection = get_mongo_collection()
        
#         # Validate ObjectId
#         try:
#             obj_id = ObjectId(connection_id)
#         except:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Invalid connection_id format"
#             )
        
#         # Find connection
#         connection = collection.find_one({"_id": obj_id})
#         if not connection:
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail=f"Connection with ID '{connection_id}' not found"
#             )
        
#         # Convert ObjectId to string
#         connection["_id"] = str(connection["_id"])
#         connection["created_at"] = connection["created_at"].isoformat()
#         connection["updated_at"] = connection["updated_at"].isoformat()
        
#         # Remove sensitive data if not requested
#         if not include_credentials:
#             if "sql_credentials" in connection:
#                 connection["sql_credentials"].pop("password", None)
#             if "nosql_credentials" in connection:
#                 # Mask connection URL
#                 if "connection_url" in connection["nosql_credentials"]:
#                     url = connection["nosql_credentials"]["connection_url"]
#                     if "@" in url:
#                         parts = url.split("@")
#                         connection["nosql_credentials"]["connection_url"] = f"***@{parts[-1]}"
        
#         return {
#             "success": True,
#             "connection": connection
#         }
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error retrieving connection: {str(e)}"
#         )


# @router.put("/{connection_id}")
# async def update_connection(connection_id: str, request: SaveConnectionRequest):
#     """
#     Update an existing database connection
    
#     Args:
#         connection_id: MongoDB ObjectId of the connection
#         request: SaveConnectionRequest with updated details
        
#     Returns:
#         Update status
#     """
#     try:
#         collection = get_mongo_collection()
        
#         # Validate ObjectId
#         try:
#             obj_id = ObjectId(connection_id)
#         except:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Invalid connection_id format"
#             )
        
#         # Check if connection exists
#         existing = collection.find_one({"_id": obj_id})
#         if not existing:
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail=f"Connection with ID '{connection_id}' not found"
#             )
        
#         # Check if new name conflicts with another connection
#         if request.connection_name != existing["connection_name"]:
#             name_conflict = collection.find_one({
#                 "connection_name": request.connection_name,
#                 "_id": {"$ne": obj_id}
#             })
#             if name_conflict:
#                 raise HTTPException(
#                     status_code=status.HTTP_409_CONFLICT,
#                     detail=f"Connection with name '{request.connection_name}' already exists"
#                 )
        
#         # Prepare update document
#         update_doc = {
#             "connection_name": request.connection_name,
#             "connection_type": request.connection_type,
#             "description": request.description,
#             "is_active": request.is_active,
#             "updated_at": datetime.utcnow()
#         }
        
#         # Add credentials based on type
#         if request.connection_type == "sql":
#             update_doc["sql_credentials"] = request.sql_credentials.dict()
#             update_doc["db_type"] = request.sql_credentials.db_type
#             # Remove nosql_credentials if switching types
#             collection.update_one({"_id": obj_id}, {"$unset": {"nosql_credentials": ""}})
#         else:  # nosql
#             update_doc["nosql_credentials"] = request.nosql_credentials.dict()
#             update_doc["db_type"] = request.nosql_credentials.db_type
#             # Remove sql_credentials if switching types
#             collection.update_one({"_id": obj_id}, {"$unset": {"sql_credentials": ""}})
        
#         # Update in MongoDB
#         collection.update_one({"_id": obj_id}, {"$set": update_doc})
        
#         return {
#             "success": True,
#             "message": f"Connection '{request.connection_name}' updated successfully",
#             "connection_id": connection_id
#         }
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error updating connection: {str(e)}"
#         )


# @router.delete("/{connection_id}", response_model=DeleteConnectionResponse)
# async def delete_connection(connection_id: str):
#     """
#     Delete a database connection
    
#     Args:
#         connection_id: MongoDB ObjectId of the connection
        
#     Returns:
#         DeleteConnectionResponse with deletion status
#     """
#     try:
#         collection = get_mongo_collection()
        
#         # Validate ObjectId
#         try:
#             obj_id = ObjectId(connection_id)
#         except:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Invalid connection_id format"
#             )
        
#         # Check if connection exists
#         existing = collection.find_one({"_id": obj_id})
#         if not existing:
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail=f"Connection with ID '{connection_id}' not found"
#             )
        
#         # Delete from MongoDB
#         collection.delete_one({"_id": obj_id})
        
#         return DeleteConnectionResponse(
#             success=True,
#             message=f"Connection '{existing['connection_name']}' deleted successfully",
#             connection_id=connection_id
#         )
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error deleting connection: {str(e)}"
#         )


# @router.post("/{connection_id}/test")
# async def test_connection(connection_id: str):
#     """
#     Test a database connection
    
#     Args:
#         connection_id: MongoDB ObjectId of the connection
        
#     Returns:
#         Test result
#     """
#     try:
#         collection = get_mongo_collection()
        
#         # Validate ObjectId
#         try:
#             obj_id = ObjectId(connection_id)
#         except:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Invalid connection_id format"
#             )
        
#         # Find connection
#         connection = collection.find_one({"_id": obj_id})
#         if not connection:
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail=f"Connection with ID '{connection_id}' not found"
#             )
        
#         # Test connection based on type
#         if connection["connection_type"] == "sql":
#             from sqlalchemy import create_engine
#             creds = connection["sql_credentials"]
            
#             # Build connection string
#             if creds["db_type"] == "postgresql":
#                 conn_str = f"postgresql://{creds['username']}:{creds['password']}@{creds['host']}:{creds['port']}/{creds['database']}"
#             elif creds["db_type"] == "mysql":
#                 conn_str = f"mysql+pymysql://{creds['username']}:{creds['password']}@{creds['host']}:{creds['port']}/{creds['database']}"
#             elif creds["db_type"] == "sqlite":
#                 conn_str = f"sqlite:///{creds['database']}"
#             else:
#                 return {"success": False, "message": f"Testing not yet implemented for {creds['db_type']}"}
            
#             engine = create_engine(conn_str)
#             with engine.connect() as conn:
#                 result = conn.execute("SELECT 1")
#                 result.fetchone()
            
#             return {"success": True, "message": "SQL connection test successful"}
        
#         else:  # nosql
#             creds = connection["nosql_credentials"]
            
#             if creds["db_type"] == "mongodb":
#                 test_client = MongoClient(creds["connection_url"], serverSelectionTimeoutMS=5000)
#                 test_client.server_info()  # Test connection
#                 test_client.close()
#                 return {"success": True, "message": "MongoDB connection test successful"}
#             else:
#                 return {"success": False, "message": f"Testing not yet implemented for {creds['db_type']}"}
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         return {
#             "success": False,
#             "message": f"Connection test failed: {str(e)}"
#         }
