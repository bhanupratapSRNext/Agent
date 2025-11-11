# from Connections.Mongo_con import client
from config.mongo_con import client

from werkzeug.security import generate_password_hash, check_password_hash

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr
import jwt
import os
import datetime

# Create FastAPI router
router = APIRouter(prefix="/user_api", tags=["user"])

SECRET_KEY = os.getenv('SECRET_KEY')

# MongoDB setup
collection = client["chat-bot"]
coll = collection["User_details"]


# Pydantic models for request validation
class LoginRequest(BaseModel):
    email: str
    password: str


class SignupRequest(BaseModel):
    email: str
    password: str


@router.post("/user/login")
def user_login(data: LoginRequest):
    """User login endpoint"""
    
    # Validate required fields (automatically done by Pydantic)
    if not data.email or not data.password:
        raise HTTPException(status_code=400, detail="email and password are required")

    # Find the user in the database
    user = coll.find_one({'username': data.email})

    if user and check_password_hash(user['password'], data.password):
        token = generate_token(user['_id'])
        return {
            'message': 'Login successful', 
            'token': token, 
            'user_id': user['username']
        }
    
    raise HTTPException(status_code=401, detail="Invalid email or password")


@router.post("/user/signup")
def user_signup(data: SignupRequest):
    """User signup endpoint"""
    
    # Validate required fields (automatically done by Pydantic)
    if not data.email or not data.password:
        raise HTTPException(status_code=400, detail="email and password are required")

    # Check if the user already exists
    user = coll.find_one({'username': data.email})
    if user:
        raise HTTPException(status_code=400, detail="User already exists")

    # Hash the password before saving it
    hashed_password = generate_password_hash(data.password, method='scrypt')

    # Create a new user
    new_user = {
        'username': data.email,
        'password': hashed_password
    }
    
    # Insert into MongoDB
    coll.insert_one(new_user)
    
    return {"message": "User created successfully"}


# Helper function to generate JWT token
def generate_token(user_id):
    """Generate JWT token for authenticated user"""
    payload = {
        'user_id': str(user_id),
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')
    return token
