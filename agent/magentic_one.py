import hashlib
import time
import os
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage

# Direct OpenAI import for simple LLM calls
from openai import AsyncOpenAI

from tools.vector_pinecone import VectorRetriever
from tools.sql_postgres import SQLTool

import json
import re

class MagenticAgent:
    """Magentic-One implementation"""
    
    def __init__(self, api_key: str, max_turns: int = 2, model: str = "openai/gpt-4o-mini", memory = None):
        self.api_key = api_key
        self.max_turns = max_turns
        self.model = model
        self.base_url = os.getenv("BASE_URL")
        
        # Initialize memory (RollingMemory passed from EcommerceAgent)
        # This now handles both conversation history AND response caching
        self.memory = memory
        
        # Initialize tools
        self.vector_tool = VectorRetriever()
        self.sql_tool = SQLTool()
        
        # Initialize LLM clients
        # For direct simple calls (RAG, SQL) - Using OpenRouter
        self.openai_client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.base_url
        )
        
        # For Magentic-One orchestrator - Using OpenRouter
        # Extract base model name for model_info (remove provider prefix)
        base_model = model.split('/')[-1] if '/' in model else model
        
        from autogen_core.models import ModelInfo
        model_info = ModelInfo(
            vision=False,
            function_calling=True,
            json_output=True,
            family=base_model
        )
        
        self.llm_client = OpenAIChatCompletionClient(
            model=model,
            api_key=api_key,
            base_url=self.base_url,
            model_info=model_info
        )
        
        # Lazy initialization for Magentic-One team
        self._team = None
        
        print("✅ Magentic Agent initialized")
        print(f"  - Model: {model}")
        print(f"  - Provider: OpenRouter")
        print(f"  - Cache: {'Enabled' if memory else 'Disabled'}")
        print(f"  - Memory: {'Enabled' if memory else 'Disabled'}")
        print(f"  - Max turns: {max_turns}")
    
    def get_memory(self, session_id: str) -> List:
        """Get conversation history from memory"""
        if self.memory:
            return self.memory.get(session_id)
        return []
    
    def clear_memory(self, session_id: str):
        """Clear conversation history"""
        if self.memory:
            self.memory.clear(session_id)
    
    async def close(self):
        """Close agent and cleanup"""
        pass
    
    async def _plan_execution(self, user_query: str, conversation_history: List) -> Dict[str, Any]:
        """
        Master Planner: Single LLM call that does routing, enrichment, and execution planning.
        
        This replaces separate routing and enrichment steps with one intelligent call.
        
        Returns a plan dictionary with:
        - route: 'smalltalk', 'sql', 'rag', 'parallel', or 'sequential'
        - enriched_query: The context-enriched query
        - sql_task: (optional) For parallel execution
        - rag_task: (optional) For parallel execution
        """
        
        # Format conversation history
        history_text = ""
        if conversation_history:
            recent = conversation_history[-8:]  # Last 8 exchanges
            history_text = "\n".join([
                f"{'User' if i % 2 == 0 else 'Assistant'}: {msg}"
                for i, msg in enumerate(recent)
            ])

        from prompts.master_planner import MASTER_PLANNER_SYSTEM_PROMPT
        system_prompt = MASTER_PLANNER_SYSTEM_PROMPT

        user_message = f"""Conversation History:
            {history_text if history_text else "No previous conversation"}
            Current Query: {user_query}
            Create execution plan:"""

        try:
            response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.0,
                max_tokens=400
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                plan = json.loads(json_match.group())
                
                # Validate and set defaults
                plan.setdefault('route', 'sequential')
                plan.setdefault('enriched_query', user_query)
                plan.setdefault('sql_task', None)
                plan.setdefault('rag_task', None)
                    
                return plan
            
            # Fallback if JSON parsing fails
            print("⚠️ Master Planner: Failed to parse JSON, using fallback")
            return {
                'route': 'sequential',
                'enriched_query': user_query,
                'sql_task': None,
                'rag_task': None
            }
            
        except Exception as e:
            print(f"❌ Master Planner error: {e}, using fallback routing")
            return {
                'route': 'sequential',
                'enriched_query': user_query,
                'sql_task': None,
                'rag_task': None
            }
  
    def _init_team(self):
        """Lazy initialization of Magentic-One team"""
        if self._team is not None:
            return
        
        # Create specialized agents with callable tools
        def rag_search(query: str) -> str:
            """Search e-commerce knowledge base"""
            docs = self.vector_tool.retrieve(query, k=3)
            if not docs:
                return "No relevant documents found."
            return "\n\n".join([f"Document {i+1}:\n{doc['text']}" for i, doc in enumerate(docs)])
        
        def sql_query(question: str) -> str:
            """Execute SQL query on database"""
            result = self.sql_tool.run(question)
            if "error" in result:
                return f"Error: {result['error']}"
            rows = result.get("rows", [])
            if not rows:
                return "No data found."
            return "\n".join([str(row) for row in rows])
        
        rag_agent = AssistantAgent(
            name="RAG_Agent",
            model_client=self.llm_client,
            tools=[rag_search],
            description="Searches e-commerce documents and reports for insights"
        )
        
        sql_agent = AssistantAgent(
            name="SQL_Agent",
            model_client=self.llm_client,
            tools=[sql_query],
            description="Executes SQL queries on database"
        )
        
        # Create Magentic-One team (only with our specific agents, no web/file surfer)
        # By only including RAG and SQL agents, we prevent internet fetching
        self._team = MagenticOneGroupChat(
            participants=[rag_agent, sql_agent],
            model_client=self.llm_client,
            max_turns=self.max_turns,
            termination_condition=TextMentionTermination("TERMINATE")
        )
   
   
    async def _execute_sql(self, query: str) -> str:
        """Execute SQL query (needs LLM to generate SQL)"""
        try:
            # Use the SQL tool's run method which generates and executes SQL
            result_dict = self.sql_tool.run(query)
            
            if "error" in result_dict:
                return f"Database error: {result_dict['error']}"
            
            rows = result_dict.get("rows", [])
            
            # Format response
            if rows:
                return rows
            else:
                note = result_dict.get("note", "")
                if note:
                    return note
                return "No data found for this query."
                
        except Exception as e:
            print(f"❌ SQL execution error: {e}")
            return "I encountered an error generating or executing the SQL query."
    
    async def _execute_rag(self, query: str) -> str:
        """Execute RAG search"""
        try:
            # Search vector database
            docs = self.vector_tool.retrieve(query, k=5)
            
            if not docs:
                return "I couldn't find specific information about that in our database."
            
            # Prepare context from retrieved documents
            context = "\n\n".join([doc["text"] for doc in docs])
            
            return context
            # Generate response using LLM
        #     system_msg = (
        #     "You are a helpful, concise assistant for an e-commerce analytics product.\n"
        #     "You MUST first infer the user's intent:\n"
        #     "• answer the user's question. Use the provided Context ONLY if it is relevant. "
        #     "  If context is missing or insufficient, say so briefly and suggest one clarifying detail. "
        #     "  Never fabricate numbers or claims.\n"
        #     "Style: warm, direct, no fluff. Prefer bullet points only when helpful."
        # )
            
        #     prompt = f"""Based on these e-commerce insights, answer the question: {query}
        #     Context: {context}
        #     Provide a clear, concise answer."""
            
        #     # Use direct OpenAI client for simple calls
        #     response = await self.openai_client.chat.completions.create(
        #         model=self.model,
        #         messages=[{"role": "system", "content": system_msg},
        #                   {"role": "user", "content": prompt}]
        #     )
        #     return response.choices[0].message.content
            
        except Exception as e:
            print(f"❌ RAG execution error: {e}")
            return "I encountered an error searching our e-commerce knowledge base."
    
    async def _execute_parallel(self, query: str, sql_task: str, rag_task: str) -> str:
        """
        Execute SQL and RAG tasks in parallel and synthesize results.
        This is faster than sequential execution for independent tasks.
        """
        try:
            import asyncio
            
            # Execute both tasks concurrently
            sql_result, rag_result = await asyncio.gather(
                self._execute_sql(sql_task),
                self._execute_rag(rag_task),
                return_exceptions=True
            )
            
            # Handle errors in parallel execution
            if isinstance(sql_result, Exception):
                sql_result = f"SQL Error: {str(sql_result)}"
            if isinstance(rag_result, Exception):
                rag_result = f"RAG Error: {str(rag_result)}"
            
            # Synthesize the results using LLM
#             system_prompt = """You are a result synthesizer. Combine data from SQL and RAG sources into a clear, natural answer."""
            
#             synthesis_prompt = f"""Original Query: "{query}"

# SQL Database Result:
# {sql_result}

# RAG Knowledge Base Result:
# {rag_result}

# Synthesize these into a single, coherent answer that addresses the original query."""

#             response = await self.openai_client.chat.completions.create(
#                 model=self.model,
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": synthesis_prompt}
#                 ],
#                 temperature=0.1
#             )
            
            return sql_result + "\n\n" + rag_result
            
        except Exception as e:
            print(f"❌ Parallel execution failed: {e}. Falling back to sequential mode.")
            return await self._execute_sequential(query)
    
    async def _execute_sequential(self, query: str) -> str:
        """
        Execute complex query using the sequential Magentic-One orchestrator.
        Used for queries where tasks are dependent on each other.
        """
        print("🐌 Executing in sequential orchestrator mode")
        try:
            self._init_team()
            result = await self._team.run(task=query)
            
            if result and result.messages:
                return result.messages[-1].content
            else:
                return "I couldn't generate a complete response for this complex query."
                
        except Exception as e:
            print(f"❌ Sequential execution error: {e}")
            return "I encountered an error processing this complex query."
    
    async def query(self, user_query: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Execute query with ultra-fast routing and validation
        
        Args:
            user_query: User's query string
            session_id: Session identifier for memory management
        
        Returns:
        {
            'response': str,
            'execution_time': float,
            'route': str,
            'cached': bool
        }
        """
        start_time = time.time()
        
        # Check cache first (using memory system)
        if self.memory:
            cached_response = self.memory.get_cached_response(user_query, session_id)
            if cached_response:
                execution_time = (time.time() - start_time) * 1000
                return {
                    'response': cached_response,
                    'execution_time': execution_time,
                    'route': 'cache',
                    'cached': True
                }

        conversation_history = self.get_memory(session_id) if self.memory else []
        plan = await self._plan_execution(user_query, conversation_history)
        
        route_type = plan['route']
        enriched_query = plan['enriched_query']
        
        print(f"🎯 Route: {route_type}")
        if enriched_query != user_query:
            print(f"📝 Enriched: '{user_query}' → '{enriched_query[:80]}...'")
        
        # Execute based on the plan
        if route_type == 'smalltalk':
            response = plan['reply']
        elif route_type == 'sql':
            response = await self._execute_sql(enriched_query)
        elif route_type == 'rag':
            response = await self._execute_rag(enriched_query)
        elif route_type == 'parallel':
            # Execute SQL and RAG in parallel, then synthesize
            response = await self._execute_parallel(
                enriched_query,
                plan['sql_task'],
                plan['rag_task']
            )
        else:  # sequential
            # Complex orchestration with dependencies
            response = await self._execute_sequential(enriched_query)
        
        execution_time = (time.time() - start_time) * 1000
        
        return {
            'response': response,
            'execution_time': execution_time,
            'route': route_type,
            'cached': False
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics from memory system"""
        if self.memory:
            return self.memory.get_cache_stats()
        return {'cache': 'disabled'}
