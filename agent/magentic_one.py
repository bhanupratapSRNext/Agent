import hashlib
import time
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


# ============================================================================
# SQL TEMPLATES FOR INSTANT RESPONSES
# ============================================================================

SQL_TEMPLATES = {
    'top_products': """
SELECT product_description, SUM(sales_revenue) as total_revenue
FROM products
GROUP BY product_description
ORDER BY total_revenue DESC
LIMIT {limit};
""",
    
    'total_revenue': """
SELECT SUM(sales_revenue) as total_revenue
FROM products;
""",
    
    'product_count': """
SELECT COUNT(DISTINCT product_description) as product_count
FROM products;
""",
    
    'average_quantity': """
SELECT AVG(quantity) as avg_quantity
FROM products;
""",
    
    'revenue_by_region': """
SELECT region, SUM(sales_revenue) as total_revenue
FROM products
GROUP BY region
ORDER BY total_revenue DESC;
""",
    
    'products_by_category': """
SELECT product_category, COUNT(*) as product_count
FROM products
GROUP BY product_category
ORDER BY product_count DESC;
""",
}


def match_sql_template(query: str) -> Optional[tuple[str, Dict[str, Any]]]:
    """Match query to SQL template for instant execution"""
    query_lower = query.lower()
    
    # Top products
    if any(word in query_lower for word in ['top', 'best', 'highest']) and \
       any(word in query_lower for word in ['product', 'item', 'selling']):
        # Extract limit if specified
        limit = 5
        if '10' in query:
            limit = 10
        elif '3' in query:
            limit = 3
        return SQL_TEMPLATES['top_products'], {'limit': limit}
    
    # Total revenue
    if any(word in query_lower for word in ['total', 'overall']) and \
       any(word in query_lower for word in ['revenue', 'sales', 'income']):
        return SQL_TEMPLATES['total_revenue'], {}
    
    # Product count
    if any(word in query_lower for word in ['how many', 'count', 'number']) and \
       'product' in query_lower:
        return SQL_TEMPLATES['product_count'], {}
    
    # Average quantity
    if 'average' in query_lower and 'quantity' in query_lower:
        return SQL_TEMPLATES['average_quantity'], {}
    
    # Revenue by region
    if 'region' in query_lower and any(word in query_lower for word in ['revenue', 'sales']):
        return SQL_TEMPLATES['revenue_by_region'], {}
    
    # Products by category
    if 'category' in query_lower or 'categories' in query_lower:
        return SQL_TEMPLATES['products_by_category'], {}
    
    return None


# ============================================================================
# SMART PRE-ROUTER - FASTER THAN BEFORE
# ============================================================================

class UltraFastRouter:
    """Lightning-fast query routing with template matching"""
    
    # SQL indicators
    SQL_KEYWORDS = [
        'recommend', 'best', 'highest', 'lowest', 'total', 'average', 'count',
        'how many', 'number of', 'sum', 'revenue', 'sales', 'quantity',
        'product name', 'region', 'category', 'price', 'order'
    ]
    
    # RAG indicators
    RAG_KEYWORDS = [
        'trend', 'strategy', 'practice', 'guide', 'how to', 'what is',
        'explain', 'advice', 'recommendation', 'insight', 'report',
        'e-commerce', 'ecommerce', 'cross-border', 'market', 'customer'
    ]
    
    # Complex query indicators
    COMPLEX_KEYWORDS = [
        'both', 'and also', 'as well as', 'compare', 'versus', 'vs',
        'difference between', 'along with', 'combined with', 'together'
    ]
    
    @classmethod
    def route(cls, query: str) -> tuple[str, Optional[tuple]]:
        """
        Route query with template matching
        Returns: (route_type, template_info)
        """
        query_lower = query.lower()
        
        # First: Check if it's a complex query (needs orchestrator)
        if any(keyword in query_lower for keyword in cls.COMPLEX_KEYWORDS):
            return 'orchestrator', None
        
        # Second: Check for SQL template match (FASTEST!)
        template_match = match_sql_template(query)
        if template_match:
            return 'sql_template', template_match
        
        # Third: Check for SQL query (needs LLM to generate SQL)
        sql_score = sum(1 for keyword in cls.SQL_KEYWORDS if keyword in query_lower)
        
        # Fourth: Check for RAG query
        rag_score = sum(1 for keyword in cls.RAG_KEYWORDS if keyword in query_lower)
        
        # Decide based on scores
        if sql_score > rag_score:
            return 'sql', None
        elif rag_score > 0:
            return 'rag', None
        else:
            # Default to RAG for general questions
            return 'rag', None


# ============================================================================
# ENHANCED QUERY CACHE
# ============================================================================

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


# ============================================================================
#  MAGENTIC-ONE AGENT
# ============================================================================

class MagenticAgent:
    """Ultra-optimized Magentic-One implementation"""
    
    def __init__(self, api_key: str, enable_cache: bool = True, max_turns: int = 2):
        self.api_key = api_key
        self.enable_cache = enable_cache
        self.max_turns = max_turns
        
        # Initialize cache
        self.cache = SmartCache() if enable_cache else None
        
        # Initialize tools
        self.vector_tool = VectorRetriever()
        self.sql_tool = SQLTool()
        
        # Initialize LLM clients
        # For direct simple calls (RAG, SQL)
        self.openai_client = AsyncOpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
        
        # For Magentic-One orchestrator
        self.llm_client = OpenAIChatCompletionClient(
            model="gpt-4o-mini",
            api_key=api_key
        )
        
        # Lazy initialization for Magentic-One team
        self._team = None
        
        print("✅ Ultra-Fast Magentic Agent initialized")
        print(f"   - Cache: {'Enabled' if enable_cache else 'Disabled'}")
        print(f"   - Max turns: {max_turns}")
    
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
            """Execute SQL query on sales database"""
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
            description="Executes SQL queries on sales database"
        )
        
        # Create Magentic-One team
        self._team = MagenticOneGroupChat(
            participants=[rag_agent, sql_agent],
            model_client=self.llm_client,
            max_turns=self.max_turns,
            termination_condition=TextMentionTermination("TERMINATE")
        )
    
    async def _execute_sql_template(self, template: str, params: Dict[str, Any]) -> str:
        """Execute SQL template instantly"""
        try:
            # Format template with parameters
            sql_query = template.format(**params) if params else template
            
            # Execute query directly against database
            with self.sql_tool.engine.connect() as conn:
                from sqlalchemy import text
                result = conn.execute(text(sql_query))
                rows = [dict(r._mapping) for r in result]
            
            # Format response
            if rows:
                # Format results nicely
                response = "Based on the sales data:\n"
                for row in rows:
                    response += "\n" + ", ".join([f"{k}: {v}" for k, v in row.items()])
                return response
            else:
                return "No data found for this query."
                
        except Exception as e:
            print(f"❌ SQL template execution error: {e}")
            return "I encountered an error accessing the database."
    
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
                response = "Based on the sales data:\n"
                for row in rows:
                    response += "\n" + ", ".join([f"{k}: {v}" for k, v in row.items()])
                return response
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
                return "I couldn't find specific information about that in our e-commerce reports."
            
            # Prepare context from retrieved documents
            context = "\n\n".join([doc["text"] for doc in docs])
            
            # Generate response using LLM
            prompt = f"""Based on these e-commerce insights, answer the question: {query}

Context:
{context}

Provide a clear, concise answer."""
            
            # Use direct OpenAI client for simple calls
            response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"❌ RAG execution error: {e}")
            return "I encountered an error searching our e-commerce knowledge base."
    
    async def _execute_orchestrator(self, query: str) -> str:
        """Execute complex query with Magentic-One orchestrator"""
        try:
            # Initialize team if needed
            self._init_team()
            
            # Run team
            result = await self._team.run(task=query)
            
            # Extract final response
            if result and result.messages:
                return result.messages[-1].content
            else:
                return "I couldn't generate a complete response for this complex query."
                
        except Exception as e:
            print(f"❌ Orchestrator error: {e}")
            return "I encountered an error processing this complex query."
    
    async def query(self, user_query: str) -> Dict[str, Any]:
        """
        Execute query with ultra-fast routing
        
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
        
        # Route query
        route_type, template_info = UltraFastRouter.route(user_query)
        
        print(f"🎯 Route: {route_type}")
        
        # Execute based on route
        if route_type == 'sql_template':
            template, params = template_info
            response = await self._execute_sql_template(template, params)
        elif route_type == 'sql':
            response = await self._execute_sql(user_query)
        elif route_type == 'rag':
            response = await self._execute_rag(user_query)
        else:  # orchestrator
            response = await self._execute_orchestrator(user_query)
        
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
