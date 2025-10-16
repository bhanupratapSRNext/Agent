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

# Fast keyword checker for obvious smalltalk (avoids LLM call)
SMALLTALK_KEYWORDS = [
    'hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon',
    'good evening', 'how are you', "how's it going", "what's up",
    'thanks', 'thank you', 'bye', 'goodbye', 'see you',
    'who are you', 'what are you', 'what can you do', 'help me',
    'nice to meet you', 'pleased to meet you', 'howdy', 'hiya'
]

def is_obvious_smalltalk(query: str) -> bool:
    """Fast check for obvious smalltalk to skip LLM planning"""
    query_lower = query.lower().strip()
    return any(keyword in query_lower for keyword in SMALLTALK_KEYWORDS)


class SmartCache:
    """Enhanced caching with statistics"""
    
    def __init__(self, max_size: int = 200, ttl_seconds: int = 3600):
        self.cache: Dict[str, tuple[str, datetime]] = {}
        self.max_size = max_size
        self.ttl = timedelta(seconds=ttl_seconds)
        
        # Statistics
        self.hits = 0
        self.misses = 0
    
    def _hash_query(self, query: str) -> str:
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def get(self, query: str) -> Optional[str]:
        """Get cached response if available and not expired"""
        query_hash = self._hash_query(query)
        
        if query_hash in self.cache:
            response, timestamp = self.cache[query_hash]
            
            # Check if expired
            if datetime.now() - timestamp < self.ttl:
                self.hits += 1
                return response
            else:
                # Remove expired entry
                del self.cache[query_hash]
        
        self.misses += 1
        return None
    
    def set(self, query: str, response: str):
        """Cache a response"""
        query_hash = self._hash_query(query)
        
        # Evict oldest if at max size
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        self.cache[query_hash] = (response, datetime.now())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'size': len(self.cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': f"{hit_rate:.1f}%"
        }

class MagenticAgent:
    """Magentic-One implementation"""
    
    def __init__(self, api_key: str, enable_cache: bool = True, max_turns: int = 2, model: str = "openai/gpt-4o-mini", memory = None):
        self.api_key = api_key
        self.enable_cache = enable_cache
        self.max_turns = max_turns
        self.model = model
        self.base_url = os.getenv("BASE_URL", "https://openrouter.ai/api/v1")
        
        # Initialize memory (RollingMemory passed from EcommerceAgent)
        self.memory = memory
        
        # Initialize cache
        self.cache = SmartCache() if enable_cache else None
        
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
        print(f"  - Cache: {'Enabled' if enable_cache else 'Disabled'}")
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
            recent = conversation_history[-6:]  # Last 3 exchanges
            history_text = "\n".join([
                f"{'User' if i % 2 == 0 else 'Assistant'}: {msg}"
                for i, msg in enumerate(recent)
            ])
        
        system_prompt = """You are a Master Query Planner for an e-commerce AI assistant. Your job is to analyze user queries and create optimal execution plans.

**Available Resources:**
1. **SQL Database**: Specific product data (sales, prices, inventory, product details, orders, revenue)
2. **RAG Knowledge Base**: General e-commerce knowledge (market trends, strategies, reports, best practices)
3. **Direct Response**: For greetings and simple conversation

**Your Task:**
Analyze the user's query (with conversation history if provided) and respond with a JSON object:

```json
{
  "route": "<route_type>",
  "enriched_query": "<enriched_query>",
  "sql_task": "<sql_task or null>",
  "rag_task": "<rag_task or null>"
}
```

**Route Types:**

1. **"smalltalk"** - Greetings, thanks, casual chat
   - enriched_query: same as original
   - sql_task: null, rag_task: null

2. **"sql"** - Needs specific product/sales/inventory data from database
   - enriched_query: refined query with context from history
   - sql_task: null, rag_task: null
   - Examples: "What are our top products?", "Show me sales for SKU 123", "Which products are low in stock?"

3. **"rag"** - Needs general e-commerce knowledge/trends/strategies
   - enriched_query: refined query with context
   - sql_task: null, rag_task: null
   - Examples: "What are current e-commerce trends?", "Best practices for conversion?", "How to optimize checkout?"

4. **"parallel"** - Needs BOTH independent SQL data AND RAG knowledge (can run simultaneously)
   - enriched_query: complete refined query
   - sql_task: specific question for SQL database
   - rag_task: specific question for RAG knowledge base
   - Example: "Compare our top product's sales with market trends" → sql_task: "What is our top-selling product and its sales?", rag_task: "What are the current market trends?"

5. **"sequential"** - Complex query where tasks depend on each other (needs orchestration)
   - enriched_query: complete refined query
   - sql_task: null, rag_task: null
   - Example: "Find products with declining sales and suggest strategies" (need SQL results first to know which products, then RAG for strategies)

**Enrichment Rules:**
- If conversation history exists, add relevant context to enriched_query
- If user says "it", "that", "them", resolve the reference from history
- If query is vague (like "revenue" or "t-shirt"), add specificity from previous context
- If query is already complete, enriched_query = original query

**Examples:**

Input: "Hello!"
Output: {"route": "smalltalk", "enriched_query": "Hello!", "sql_task": null, "rag_task": null}

Input: "What are our best-selling products?"
Output: {"route": "sql", "enriched_query": "What are our best-selling products?", "sql_task": null, "rag_task": null}

Input: "What are the latest e-commerce trends?"
Output: {"route": "rag", "enriched_query": "What are the latest e-commerce trends?", "sql_task": null, "rag_task": null}

Input: "Compare our top t-shirt sales with sustainable fashion trends"
Output: {"route": "parallel", "enriched_query": "Compare our top t-shirt sales with sustainable fashion trends", "sql_task": "What is our top-selling t-shirt and what are its sales figures?", "rag_task": "What are the current trends in sustainable fashion?"}

History: "User: Show me product SKU 12345\nAssistant: [product details]"
Input: "What about its sales?"
Output: {"route": "sql", "enriched_query": "What are the sales figures for product SKU 12345?", "sql_task": null, "rag_task": null}

History: "User: Show me menswear products\nAssistant: [list of menswear]"
Input: "t-shirt"
Output: {"route": "sql", "enriched_query": "Show me t-shirt products from menswear category", "sql_task": null, "rag_task": null}

**Critical:** Output ONLY valid JSON. No explanations, no markdown blocks, just the JSON object."""

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
    
    async def _execute_smalltalk(self, query: str) -> str:
        """Execute small talk - direct, fast response without tools"""
        try:
            system_msg = """You are a friendly e-commerce assistant.

Respond naturally to greetings and casual conversation.
Keep responses brief (1-2 sentences maximum).
Be warm but professional.

If asked what you can do, mention:
- Answer questions about e-commerce trends and strategies
- Analyze sales data and product performance
- Provide insights from e-commerce reports
- Help with data-driven recommendations

Do not cite sources or mention technical details. Just be conversational."""
            
            # Direct LLM call - no tools, no context retrieval, no orchestration
            response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": query}
                ],
                max_tokens=100,  # Keep it short for greetings
                temperature=0.7  # More natural for conversation
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"❌ Small talk error: {e}")
            # Friendly fallback
            return "Hello! I'm here to help you with e-commerce insights and data analysis. What can I help you with today?"
    
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
            docs = self.vector_tool.retrieve(query, k=3)
            
            if not docs:
                return "I couldn't find specific information about that in our database."
            
            # Prepare context from retrieved documents
            context = "\n\n".join([doc["text"] for doc in docs])
            
            # Generate response using LLM
            system_msg = (
            "You are a helpful, concise assistant for an e-commerce analytics product.\n"
            "You MUST first infer the user's intent:\n"
            "• If the message is smalltalk (greetings, pleasantries, jokes, about-you), "
            "  respond naturally . Do NOT cite documents or domain facts.\n"
            "• Otherwise, answer the user's question. Use the provided Context ONLY if it is relevant. "
            "  If context is missing or insufficient, say so briefly and suggest one clarifying detail. "
            "  Never fabricate numbers or claims.\n"
            "Style: warm, direct, no fluff. Prefer bullet points only when helpful."
        )
            
            prompt = f"""Based on these e-commerce insights, answer the question: {query}
            Context: {context}
            Provide a clear, concise answer."""
            
            # Use direct OpenAI client for simple calls
            response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system_msg},
                          {"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"❌ RAG execution error: {e}")
            return "I encountered an error searching our e-commerce knowledge base."
    
    async def _execute_parallel(self, query: str, sql_task: str, rag_task: str) -> str:
        """
        Execute SQL and RAG tasks in parallel and synthesize results.
        This is faster than sequential execution for independent tasks.
        """
        print("🚀 Executing in parallel mode")
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
            system_prompt = """You are a result synthesizer. Combine data from SQL and RAG sources into a clear, natural answer."""
            
            synthesis_prompt = f"""Original Query: "{query}"

SQL Database Result:
{sql_result}

RAG Knowledge Base Result:
{rag_result}

Synthesize these into a single, coherent answer that addresses the original query."""

            response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": synthesis_prompt}
                ],
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
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
        
        # Check cache first
        if self.enable_cache:
            cached_response = self.cache.get(user_query)
            if cached_response:
                execution_time = (time.time() - start_time) * 1000
                return {
                    'response': cached_response,
                    'execution_time': execution_time,
                    'route': 'cache',
                    'cached': True
                }

        # Check for obvious smalltalk (skip LLM planning)
        if is_obvious_smalltalk(user_query):
            response = await self._execute_smalltalk(user_query)
            route_type = 'smalltalk'
        else:
            # Master Planner: Single LLM call for routing + enrichment + planning
            conversation_history = self.get_memory(session_id) if self.memory else []
            plan = await self._plan_execution(user_query, conversation_history)
            
            route_type = plan['route']
            enriched_query = plan['enriched_query']
            
            print(f"🎯 Route: {route_type}")
            if enriched_query != user_query:
                print(f"📝 Enriched: '{user_query}' → '{enriched_query[:80]}...'")
            
            # Execute based on the plan
            if route_type == 'smalltalk':
                response = await self._execute_smalltalk(user_query)
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
        
        # Cache response
        if self.enable_cache:
            self.cache.set(user_query, response)
        
        execution_time = (time.time() - start_time) * 1000
        
        return {
            'response': response,
            'execution_time': execution_time,
            'route': route_type,
            'cached': False
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if self.cache:
            return self.cache.get_stats()
        return {'cache': 'disabled'}
