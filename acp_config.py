import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class ACPConfig:
    """ACP SDK Configuration class"""
    
    # Server Configuration
    SERVER_ID = os.getenv("ACP_SERVER_ID")
    SERVER_NAME = os.getenv("ACP_SERVER_NAME")
    SERVER_VERSION = os.getenv("ACP_SERVER_VERSION")
    SERVER_PORT = int(os.getenv("SERVER_PORT"))
    UVICORN_RELOAD = os.getenv("UVICORN_RELOAD", "True").lower() == "true"
    UVICORN_LOG_LEVEL = os.getenv("UVICORN_LOG_LEVEL", "info")
    
    # ACP Features
    ENABLE_AUTHENTICATION = os.getenv("ACP_ENABLE_AUTH", "true").lower() == "true"
    ENABLE_RATE_LIMITING = os.getenv("ACP_ENABLE_RATE_LIMIT", "true").lower() == "true"
    ENABLE_LOGGING = os.getenv("ACP_ENABLE_LOGGING", "true").lower() == "true"
    
    # Rate Limiting Configuration
    RATE_LIMIT_REQUESTS_PER_MINUTE = int(os.getenv("ACP_RATE_LIMIT_RPM", "100"))
    RATE_LIMIT_BURST_SIZE = int(os.getenv("ACP_RATE_LIMIT_BURST", "20"))
    
    # Authentication Configuration
    AUTH_JWT_SECRET = os.getenv("ACP_JWT_SECRET")
    AUTH_JWT_ALGORITHM = os.getenv("ACP_JWT_ALGORITHM", "HS256")
    AUTH_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACP_TOKEN_EXPIRE", "60"))
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("ACP_LOG_LEVEL", "INFO")
    LOG_FORMAT = os.getenv("ACP_LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Agent Configuration
    AGENT_MAX_CONTEXT_LENGTH = int(os.getenv("AGENT_MAX_CONTEXT", "4000"))
    AGENT_TIMEOUT_SECONDS = int(os.getenv("AGENT_TIMEOUT", "30"))
    AGENT_RETRY_ATTEMPTS = int(os.getenv("AGENT_RETRY_ATTEMPTS", "3"))
    
    # Business Logic Configuration (preserved from your original)
    DEFAULT_TOP_K = int(os.getenv("TOP_K", "5"))
    DEFAULT_RELEVANCE = float(os.getenv("RELEVANCE_THRESHOLD", "0.55"))
    MEMORY_WINDOW = int(os.getenv("MEMORY_WINDOW", "10"))
    
    # Database Configuration (for run storage)
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./acp_runs.db")
    ENABLE_RUN_PERSISTENCE = os.getenv("ACP_ENABLE_PERSISTENCE", "true").lower() == "true"
    
    # CORS Configuration
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    CORS_ALLOW_CREDENTIALS = os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() == "true"
    
    @classmethod
    def get_acp_client_config(cls) -> Dict[str, Any]:
        """Get ACP client configuration"""
        return {
            "server_id": cls.SERVER_ID,
            "server_name": cls.SERVER_NAME,
            "server_version": cls.SERVER_VERSION,
            "enable_authentication": cls.ENABLE_AUTHENTICATION,
            "enable_rate_limiting": cls.ENABLE_RATE_LIMITING,
            "enable_logging": cls.ENABLE_LOGGING,
            "rate_limit_requests_per_minute": cls.RATE_LIMIT_REQUESTS_PER_MINUTE,
            "rate_limit_burst_size": cls.RATE_LIMIT_BURST_SIZE,
            "auth_jwt_secret": cls.AUTH_JWT_SECRET,
            "auth_jwt_algorithm": cls.AUTH_JWT_ALGORITHM,
            "auth_token_expire_minutes": cls.AUTH_TOKEN_EXPIRE_MINUTES,
            "log_level": cls.LOG_LEVEL,
            "log_format": cls.LOG_FORMAT
        }
    
    @classmethod
    def get_agent_config(cls) -> Dict[str, Any]:
        """Get agent configuration"""
        return {
            "max_context_length": cls.AGENT_MAX_CONTEXT_LENGTH,
            "timeout_seconds": cls.AGENT_TIMEOUT_SECONDS,
            "retry_attempts": cls.AGENT_RETRY_ATTEMPTS,
            "default_top_k": cls.DEFAULT_TOP_K,
            "default_relevance": cls.DEFAULT_RELEVANCE,
            "memory_window": cls.MEMORY_WINDOW
        }
    
    @classmethod
    def get_cors_config(cls) -> Dict[str, Any]:
        """Get CORS configuration"""
        return {
            "allow_origins": ["*"], 
            "allow_credentials": cls.CORS_ALLOW_CREDENTIALS,
            "allow_methods": ["*"],
            "allow_headers": ["*"]
        }
