import os
from typing import Dict, Any
from openai import AsyncOpenAI


class ResponseBuilder:
    """
    LLM-powered response builder that enhances raw agent outputs
    for clarity, professionalism, and user-friendliness.
    """
    
    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize ResponseBuilder
        
        Args:
            api_key: OpenRouter API key (defaults to env OPEN_ROUTER_KEY)
            model: Model to use (defaults to env OPENROUTER_MODEL)
        """
        self.api_key = api_key or os.getenv("OPEN_ROUTER_KEY")
        if not self.api_key:
            raise ValueError("OPEN_ROUTER_KEY required for ResponseBuilder")
        
        self.model = model or os.getenv("OPENROUTER_MODEL")
        self.base_url = os.getenv("BASE_URL")
        
        # Initialize LLM client
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
    
    async def build_response(
        self, 
        user_query: str, 
        raw_response: str, 
        route: str,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Build a polished, user-friendly response from raw agent output
        
        Args:
            user_query: Original user question
            raw_response: Raw response from agent/tool
            route: Which route was used ('sql', 'rag', 'orchestrator', 'cache')
            metadata: Optional metadata about execution
            
        Returns:
            Enhanced, user-friendly response string
        """
        # Don't enhance cached responses (already processed)
        if route == 'cache':
            return raw_response
        
        # Build system prompt based on route type
        system_prompt = self._get_system_prompt(route)
        
        # Build user prompt with context
        user_prompt = self._build_user_prompt(user_query, raw_response, route, metadata)
        
        try:
            # Call LLM to enhance response
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Low temperature for consistent, factual responses
                max_tokens=800
            )
            
            enhanced_response = completion.choices[0].message.content.strip()
            return enhanced_response
            
        except Exception as e:
            print(f"⚠️ ResponseBuilder error: {e}")
            # Fallback to raw response if enhancement fails
            return raw_response
    
    def _get_system_prompt(self, route: str) -> str:
        """Get route-specific system prompt"""
        
        base_instructions = """You are a professional e-commerce assistant. Your job is to take raw data or information and present it clearly to users.

CRITICAL RULES:
- NEVER fabricate data, numbers, or facts not present in the raw response
- If raw data is empty/missing, acknowledge it politely
- Be concise but complete
- Use natural, conversational language
- Format lists and data for easy scanning"""

        route_specific = {
            'sql': """
The raw response contains database query results with sales/product data.
- Present numbers and metrics clearly
- Use bullet points for multiple items
- Highlight key findings
- Add brief context when helpful (e.g., "Based on sales data...")""",
            
            'rag': """
The raw response contains information from e-commerce documents/reports.
- Synthesize information naturally
- Cite insights without being verbose
- Use paragraphs for concepts, bullets for lists
- Add "According to..." or "Based on..." when appropriate""",
            
            'orchestrator': """
The raw response combines multiple data sources (database + documents).
- Integrate both types of information seamlessly
- Present a cohesive answer
- Use sections if combining distinct insights
- Make connections between data and context clear"""
        }
        
        return base_instructions + "\n" + route_specific.get(route, route_specific['rag'])
    
    def _build_user_prompt(
        self, 
        user_query: str, 
        raw_response: str, 
        route: str,
        metadata: Dict[str, Any]
    ) -> str:
        """Build the user prompt with context"""
        
        prompt = f"""USER QUESTION:
{user_query}

RAW RESPONSE FROM SYSTEM:
{raw_response}

ROUTE USED: {route}
"""
        
        if metadata:
            exec_time = metadata.get('execution_time_ms', 0)
            if exec_time:
                prompt += f"EXECUTION TIME: {exec_time:.0f}ms\n"
        
        prompt += """
YOUR TASK:
Transform the raw response into a clear, professional answer that directly addresses the user's question.
Preserve ALL factual information and numbers exactly as given.
Make it easy to read and understand.

ENHANCED RESPONSE:"""
        
        return prompt
    
    async def build_error_response(self, error_message: str, user_query: str) -> str:
        """
        Build a user-friendly error response
        
        Args:
            error_message: Technical error message
            user_query: Original user question
            
        Returns:
            User-friendly error message
        """
        system_prompt = """You are a helpful assistant. Convert technical error messages into friendly, actionable responses for users.
- Be apologetic but brief
- Suggest what the user might try instead
- Don't expose technical details"""
        
        user_prompt = f"""USER ASKED: {user_query}

TECHNICAL ERROR: {error_message}

Provide a friendly, helpful error message:"""
        
        try:
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                max_tokens=150
            )
            
            return completion.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"⚠️ Error response builder failed: {e}")
            return f"I apologize, but I encountered an error processing your request. Please try rephrasing your question or contact support if the issue persists."


# Singleton instance for reuse
_response_builder_instance = None

def get_response_builder(api_key: str = None, model: str = None) -> ResponseBuilder:
    """Get or create singleton ResponseBuilder instance"""
    global _response_builder_instance
    if _response_builder_instance is None:
        _response_builder_instance = ResponseBuilder(api_key, model)
    return _response_builder_instance
