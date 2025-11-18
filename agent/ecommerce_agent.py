import uuid
import os
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime

# Import Optimized Magentic-One implementation
from agent.magentic_one import MagenticAgent

# Import ResponseBuilder for enhanced outputs
from agent.response_builder import ResponseBuilder

# Import ACP configuration
from acp_config import ACPConfig


class EcommerceAgent:
    """
    ACP-compliant E-commerce Agent powered by Microsoft's Magentic-One
    
    This agent uses Magentic-One's orchestrator to automatically route queries
    to specialized agents (RAG and SQL) without manual intent classification.
    """
    
    def __init__(self, memory=None):
        self.name = "ecommerce-magentic-one"
        self.description = "Intelligent e-commerce agent powered by Microsoft's Magentic-One multi-agent orchestrator"
        self.agent_config = ACPConfig.get_agent_config()
        self._run_cache = {}  # Simple in-memory cache for runs
        
        self.api_key = os.getenv("OPEN_ROUTER_KEY")
        if not self.api_key:
            raise ValueError("OPEN_ROUTER_KEY environment variable is required")

        # Get model from environment or use default
        self.model = os.getenv("OPENROUTER_MODEL")
        
        # Store memory instance for conversation context
        self.memory = memory
        
        # Initialize the Magentic-One agent system
        self.magentic_agent = None  # Will be initialized async
        
        # Initialize response builder for enhanced outputs
        self.response_builder = ResponseBuilder(
            api_key=self.api_key,
            model=self.model
        )
    
    async def _ensure_agent_initialized(self):
        """Lazy initialization of Ultra-Fast Magentic-One agent (async)"""
        if self.magentic_agent is None:
            self.magentic_agent = MagenticAgent(
                api_key=self.api_key,
                model=self.model,
                max_turns=2,  # Reduced for faster responses
                memory=self.memory  # Pass memory for context-aware validation and caching
            )
    
    async def execute_agent(self, request, session_id):
        """
        Execute the agent directly with request data using Magentic-One
        
        Args:
            request: The request object containing user input
            session_id: Session identifier for memory management
            
        Returns:
            Response dictionary in ACP format
        """
        try:
            # Ensure agent is initialized
            await self._ensure_agent_initialized()
            
            # Extract user input from request
            user_text = request.input[0].parts[0].content if request.input and request.input[0].parts else ""
            
            if not user_text.strip():
                raise ValueError("Empty user input provided")
            
            print(f"[Agent] Processing query: '{user_text}' for session: {session_id}")

            tenant_id = request.tenantID
            # Store user query in memory
            if self.memory:
                self.memory.append(session_id, "user", user_text)
            
            # Use Magentic-One to automatically orchestrate the agents
            result = await self.magentic_agent.query(user_text,tenant_id, session_id)
            
            # Extract raw response and metadata
            raw_response = result["response"]
            route = result.get("route", "unknown")
            cached = result.get("cached", False)
            execution_time = result.get("execution_time", 0)
            
            # Build enhanced response (skip if cached OR smalltalk - already clean)
            if cached or route == 'smalltalk':
                answer = raw_response
            else:
                answer = await self.response_builder.build_response(
                    user_query=user_text,
                    raw_response=raw_response,
                    route=route,
                    metadata={
                        "execution_time_ms": execution_time,
                        "cached": cached
                    }
                )
            
            # Store assistant response in memory
            if self.memory:
                self.memory.append(session_id, "assistant", answer)
            
            # Cache the final formatted response (only if not already cached)
            if not cached and self.memory:
                self.memory.cache_response(user_text, answer, session_id)
            
            metadata = {
                "execution_time_ms": execution_time,
                "route": route,
                "cached": cached
            }
            
            # Create routing info for metadata
            routing_info = {
                "session_id": session_id,
                "timestamp": self._now_iso(),
                "orchestrator": "MagenticOne",
                "agent_config": {
                    "top_k": self.agent_config["default_top_k"],
                    "relevance_threshold": self.agent_config["default_relevance"]
                },
                "metadata": metadata
            }
            
            # Create response in ACP format
            response = {
                "run_id": str(uuid.uuid4()),
                "agent_name": self.name,
                "session_id": session_id,
                "status": "completed",
                "output": [
                    {
                        "role": "agent/magentic-one",
                        "parts": [
                            {
                                "content_type": "text/plain",
                                "content": answer,
                                "metadata": routing_info
                            }
                        ]
                    }
                ],
                "created_at": self._now_iso(),
                "finished_at": self._now_iso()
            }
            return response
            
        except Exception as e:
            # Build friendly error message
            error_response = await self.response_builder.build_error_response(
                error_message=str(e),
                user_query=user_text if 'user_text' in locals() else "your request"
            )
            
            return {
                "run_id": str(uuid.uuid4()),
                "agent_name": self.name,
                "session_id": session_id,
                "status": "failed",
                "error": str(e),
                "output": [
                    {
                        "role": "agent/error",
                        "parts": [
                            {
                                "content_type": "text/plain",
                                "content": error_response,
                                "metadata": {"error": str(e)}
                            }
                        ]
                    }
                ],
                "created_at": self._now_iso(),
                "finished_at": self._now_iso()
            }
    
    def get_memory(self, session_id: str) -> List:
        """
        Get conversation history for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of conversation turns
        """
        if self.magentic_agent:
            return self.magentic_agent.get_memory(session_id)
        return []
    
    def clear_memory(self, session_id: str):
        """
        Clear conversation history for a session
        
        Args:
            session_id: Session identifier
        """
        if self.magentic_agent:
            self.magentic_agent.clear_memory(session_id)
    
    async def close(self):
        """Close the agent and clean up resources"""
        if self.magentic_agent:
            await self.magentic_agent.close()
    
    def _now_iso(self):
        """Get current timestamp in ISO format"""
        return datetime.utcnow().isoformat() + "Z"
    
    def get_capabilities(self):
        """
        Get agent capabilities (for ACP)
        
        Returns:
            Dictionary of agent capabilities
        """
        return {
            "name": self.name,
            "description": self.description,
            "version": "2.0.0",
            "orchestrator": "Microsoft Magentic-One",
            "features": [
                "Automatic query routing",
                "Multi-agent orchestration",
                "RAG (Retrieval Augmented Generation)",
                "SQL database queries",
                "Conversation memory",
                "Streaming responses"
            ],
            "agents": [
                {
                    "name": "EcommerceInfoAgent",
                    "description": "Handles e-commerce trends, strategies, and best practices using document retrieval"
                },
                {
                    "name": "EcommerceDataAgent",
                    "description": "Handles sales data, product performance, and database queries"
                }
            ]
        }


# For backward compatibility
_agent_instance = None

def get_agent_instance():
    """Get or create the singleton agent instance"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = EcommerceAgent()
    return _agent_instance
