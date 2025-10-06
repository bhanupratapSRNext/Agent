import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime

# Import your existing business logic (preserved)
from agent.memory import RollingMemory
from agent.tools.vector_pinecone import VectorRetriever
from agent.tools.sql_postgres import SQLTool
from agent.intent import classify_intent
from agent.response_builder import build_answer

# Import ACP configuration
from acp_config import ACPConfig


class EcommerceAgent:
    """
    ACP-compliant E-commerce Agent
    Preserves all your existing business logic while leveraging ACP SDK features
    """
    
    def __init__(self):
        self.name = "ecommerce-router"
        self.description = "Intelligent e-commerce agent that routes queries to RAG or SQL handlers"
        self.agent_config = ACPConfig.get_agent_config()
        self._run_cache = {}  # Simple in-memory cache for runs
    
    async def execute_direct(self, request, session_id):
        """
        Execute the agent directly with request data (bypassing ACP SDK Run object issues)
        """
        try:
            # Extract user input directly from request
            user_text = request.input[0].parts[0].content if request.input and request.input[0].parts else ""
            
            if not user_text.strip():
                raise ValueError("Empty user input provided")
            
            # Preserve your existing intent classification logic
            intent = classify_intent(user_text)
            
            # Preserve your existing routing logic with enhanced metadata
            routing_info = {
                "intent": intent,
                "session_id": session_id,
                "timestamp": self._now_iso(),
                "agent_config": {
                    "top_k": self.agent_config["default_top_k"],
                    "relevance_threshold": self.agent_config["default_relevance"]
                }
            }
            
            # Route based on intent (preserved from your original logic)
            if intent == "Ecommerce_Data":
                chosen_agent = "ecommerce-data"
                answer = await self._run_ecommerce_data(session_id, user_text)
            elif intent == "Ecommerce_Info":
                chosen_agent = "ecommerce-info"
                answer = await self._run_ecommerce_info(session_id, user_text)
            else:
                # Fallback logic (preserved)
                chosen_agent = "ecommerce-info"
                routing_info["note"] = "fallback_default_info"
                answer = await self._run_ecommerce_info(session_id, user_text)
            
            routing_info["routed_to"] = chosen_agent
            routing_info["confidence"] = 0.9 if intent != "Unknown" else 0.5
            
            # Update memory (preserved logic)
            from agent.memory import RollingMemory
            memory = RollingMemory(window_size=ACPConfig.MEMORY_WINDOW)
            memory.append(session_id, "user", user_text)
            memory.append(session_id, "assistant", answer)
            
            # Create response in the format your frontend expects
            response = {
                "run_id": str(uuid.uuid4()),
                "agent_name": "ecommerce-router",
                "session_id": session_id,
                "status": "completed",
                "output": [
                    {
                        "role": f"agent/{chosen_agent}",
                        "parts": [
                            {
                                "content_type": "text/plain",
                                "content": answer,
                                "metadata": {
                                    "intent": intent,
                                    "agent": chosen_agent,
                                    "confidence": routing_info["confidence"]
                                }
                            }
                        ]
                    }
                ],
                "created_at": self._now_iso(),
                "finished_at": self._now_iso()
            }
            
            return response
            
        except Exception as e:
            # Error handling
            return {
                "run_id": str(uuid.uuid4()),
                "agent_name": "ecommerce-router",
                "session_id": session_id,
                "status": "failed",
                "error": {
                    "code": "AGENT_EXECUTION_ERROR",
                    "message": str(e),
                    "details": {
                        "agent": "ecommerce-router",
                        "session_id": session_id,
                        "timestamp": self._now_iso()
                    }
                },
                "created_at": self._now_iso(),
                "finished_at": self._now_iso()
            }

    async def execute(self, run):
        """
        Execute the agent with full ACP compliance and preserved business logic
        """
        try:
            # Update run status to in-progress
            run.status = run.status.IN_PROGRESS
            
            # Extract and validate input
            user_text = self._extract_user_input(run)
            session_id = run.session_id or str(uuid.uuid4())
            
            if not user_text.strip():
                raise ValueError("Empty user input provided")
            
            # Preserve your existing intent classification logic
            intent = classify_intent(user_text)
            
            # Preserve your existing routing logic with enhanced metadata
            routing_info = {
                "intent": intent,
                "session_id": session_id,
                "timestamp": self._now_iso(),
                "agent_config": {
                    "top_k": self.agent_config["default_top_k"],
                    "relevance_threshold": self.agent_config["default_relevance"]
                }
            }
            
            # Route based on intent (preserved from your original logic)
            if intent == "Ecommerce_Data":
                chosen_agent = "ecommerce-data"
                answer = await self._run_ecommerce_data(session_id, user_text)
            elif intent == "Ecommerce_Info":
                chosen_agent = "ecommerce-info"
                answer = await self._run_ecommerce_info(session_id, user_text)
            else:
                # Fallback logic (preserved)
                chosen_agent = "ecommerce-info"
                routing_info["note"] = "fallback_default_info"
                answer = await self._run_ecommerce_info(session_id, user_text)
            
            routing_info["routed_to"] = chosen_agent
            routing_info["confidence"] = 0.9 if intent != "Unknown" else 0.5
            
            # Update memory (preserved logic)
            from agent.memory import RollingMemory
            memory = RollingMemory(window_size=ACPConfig.MEMORY_WINDOW)
            memory.append(session_id, "user", user_text)
            memory.append(session_id, "assistant", answer)
            
            # Create ACP-compliant response with enhanced metadata
            from acp_sdk import Message, MessagePart
            import json
            
            output_messages = [
                Message(
                    role=f"agent/{chosen_agent}",
                    parts=[
                        MessagePart(
                            content_type="text/plain",
                            content=answer,
                            metadata={
                                "intent": intent,
                                "agent": chosen_agent,
                                "confidence": routing_info["confidence"],
                                "processing_time": self._calculate_processing_time(run.created_at)
                            }
                        )
                    ]
                ),
                Message(
                    role=f"agent/{chosen_agent}",
                    parts=[
                        MessagePart(
                            name="routing_metadata",
                            content_type="application/json",
                            content=json.dumps(routing_info)
                        )
                    ]
                )
            ]
            
            # Update run with results
            run.status = run.status.COMPLETED
            run.output = output_messages
            run.finished_at = self._now_iso()
            
            # Cache the run
            self._run_cache[run.run_id] = run
            
            return run
            
        except Exception as e:
            # Comprehensive error handling
            run.status = run.status.FAILED
            run.error = {
                "code": "AGENT_EXECUTION_ERROR",
                "message": str(e),
                "details": {
                    "agent": "ecommerce-router",
                    "session_id": getattr(run, 'session_id', None),
                    "timestamp": self._now_iso(),
                    "retry_attempts": getattr(run, 'retry_attempts', 0)
                }
            }
            run.finished_at = self._now_iso()
            
            # Cache failed run for debugging
            self._run_cache[run.run_id] = run
            
            return run
    
    def _extract_user_input(self, run):
        """Extract user input from ACP run with validation"""
        print(f"DEBUG: Run object: {run}")
        print(f"DEBUG: Run attributes: {dir(run)}")
        
        if not hasattr(run, 'input') or not run.input:
            print("DEBUG: No input attribute or empty input")
            return ""
        
        print(f"DEBUG: Input: {run.input}")
        first_message = run.input[0]
        print(f"DEBUG: First message: {first_message}")
        print(f"DEBUG: First message attributes: {dir(first_message)}")
        
        if not hasattr(first_message, 'parts') or not first_message.parts:
            print("DEBUG: No parts attribute or empty parts")
            return ""
        
        print(f"DEBUG: Parts: {first_message.parts}")
        content = first_message.parts[0].content or ""
        print(f"DEBUG: Extracted content: '{content}'")
        return content
    
    async def _run_ecommerce_info(self, user_id: str, text: str) -> str:
        """
        Preserved RAG logic for e-commerce info queries
        Enhanced with async support and optimal error handling
        """
        try:
            from agent.memory import RollingMemory
            memory = RollingMemory(window_size=ACPConfig.MEMORY_WINDOW)
            mem = memory.get(user_id)
            retriever = VectorRetriever()
            results = retriever.retrieve(text, k=self.agent_config["default_top_k"])
            good = [r for r in results if (1.0 - r["score"]) >= self.agent_config["default_relevance"] or r["score"] >= self.agent_config["default_relevance"]]
            rag_context = [r["text"] for r in (good or results)]
            
            if not rag_context:
                return "I couldn't find relevant documents for that query. Please try rephrasing your question or ask about a different topic."
            
            return build_answer(text, mem, rag_context, sql_rows=None)
            
        except Exception as e:
            return f"I encountered an error while searching for information: {str(e)}. Please try again."
    
    async def _run_ecommerce_data(self, user_id: str, text: str) -> str:
        """
        Preserved SQL logic for e-commerce data queries
        Enhanced with async support and optimal error handling
        """
        try:
            from agent.memory import RollingMemory
            memory = RollingMemory(window_size=ACPConfig.MEMORY_WINDOW)
            mem = memory.get(user_id)
            sql_tool = SQLTool()
            out = sql_tool.run(text)
            sql_rows = out.get("rows", [])
            rag_context = None
            
            if not sql_rows:
                # Fallback to RAG if no SQL results
                retriever = VectorRetriever()
                results = retriever.retrieve(text, k=self.agent_config["default_top_k"])
                good = [r for r in results if (1.0 - r["score"]) >= self.agent_config["default_relevance"] or r["score"] >= self.agent_config["default_relevance"]]
                rag_context = [r["text"] for r in (good or results)]
            
            if not (sql_rows or rag_context):
                return "No relevant data found in the database or knowledge base. Please check your query or try a different approach."
            
            return build_answer(text, mem, rag_context, sql_rows)
            
        except Exception as e:
            return f"I encountered an error while querying the database: {str(e)}. Please try rephrasing your question."
    
    def _calculate_processing_time(self, created_at: str) -> float:
        """Calculate processing time in seconds"""
        try:
            created_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            current_time = datetime.utcnow()
            return (current_time - created_time).total_seconds()
        except:
            return 0.0
    
    def _now_iso(self) -> str:
        """Get current timestamp in ISO format"""
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    
    def get_run(self, run_id: str):
        """Get cached run by ID"""
        return self._run_cache.get(run_id)
    
    def get_all_runs(self):
        """Get all cached runs"""
        return self._run_cache.copy()
