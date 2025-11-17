import os
import json
import re
from typing import Dict, Any, Optional
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
        raw_response: Any,  # Changed from str to Any to accept lists too
        route: str,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Build a polished, user-friendly response from raw agent output
        
        Args:
            user_query: Original user question
            raw_response: Raw response from agent/tool (can be str, list, or dict)
            route: Which route was used ('sql', 'rag', 'orchestrator', 'cache')
            metadata: Optional metadata about execution
            
        Returns:
            Enhanced, user-friendly response string or JSON string with product details
        """
        # Convert raw_response to string if it's a list or other type
        if isinstance(raw_response, (list, dict)):
            raw_response = str(raw_response)
        elif not isinstance(raw_response, str):
            raw_response = str(raw_response)
        
        # Don't enhance cached responses (already processed)
        if route == 'cache':
            return raw_response
        
        # Check if response contains product details and should be formatted as JSON
        if self._contains_product_details(raw_response):
            formatted_json = await self._format_product_details_as_json(
                user_query, raw_response, route
            )
            if formatted_json:
                return formatted_json
            else:
                # If JSON formatting failed but we detected product data,
                # try a simple fallback conversion before falling back to raw response
                print(f"⚠️ Product JSON formatting failed, trying simple fallback...")
                simple_json = self._simple_json_fallback(raw_response, user_query)
                if simple_json:
                    print(f"✅ Simple fallback JSON conversion successful")
                    return simple_json
                else:
                    print(f"⚠️ All JSON formatting attempts failed, returning raw response to preserve structure")
                    return raw_response
        else:
            print(f"ℹ️ No product details detected, proceeding with standard text enhancement")
        
        # Standard response enhancement for non-product queries
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
    
    def _contains_product_details(self, raw_response: str) -> bool:
        """
        Detect if the raw response contains product details that should be formatted as JSON
        
        Returns:
            True if response contains structured product data
        """
        # Ensure raw_response is a string
        if not isinstance(raw_response, str):
            raw_response = str(raw_response)
        
        # Check for common product-related patterns
        product_indicators = [
            r'product[_\s]+(?:id|sku|name|title)',
            r'(?:price|cost|revenue)',
            r'(?:quantity|stock|inventory)',
            r'(?:category|type|brand)',
            r'sales[_\s]+(?:figure|data|amount)',
            r'\d+\s*(?:items?|products?|skus?)',
            # SQL-like output patterns
            r'\(\d+,\s*["\']',  # Tuple-like data (1, 'Product Name', ...)
            r'\[.*?\].*?\[.*?\]',  # List of lists
            # Check if it looks like tabular data
            r'(?:\w+:\s*\d+.*?\n){3,}',  # Multiple key:value pairs
            # PostgreSQL-style dict output
            r'\{["\'](?:product_|sku|price|quantity)',  # Dict with product keys
            # List of dicts pattern
            r'\[\s*\{.*?product.*?\}',  # List containing dicts with product info
        ]
        
        response_lower = raw_response.lower()
        
        # Count matching patterns
        match_count = sum(1 for pattern in product_indicators if re.search(pattern, response_lower))
        
        # Also check for specific SQL result patterns
        # Look for patterns like [{'product_id': 1, 'name': '...'}]
        dict_list_pattern = r'\[\s*\{[^}]*(?:product|sku|price|sales|revenue|quantity)[^}]*\}'
        if re.search(dict_list_pattern, response_lower):
            match_count += 2  # Strong indicator
        
        # If multiple product indicators are present, it's likely product data
        return match_count >= 2
    
    async def _format_product_details_as_json(
        self, 
        user_query: str, 
        raw_response: Any,  # Can be str, list, or dict
        route: str
    ) -> Optional[str]:
        """
        Format product details from raw response into structured JSON
        
        Args:
            user_query: Original user question
            raw_response: Raw response with product data (str, list, or dict)
            route: Route used
            
        Returns:
            JSON string with formatted product details or None if formatting fails
        """
        # Ensure raw_response is a string
        if not isinstance(raw_response, str):
            raw_response = str(raw_response)
        system_prompt = """You are a Data Formatter Specialist AI. Your primary role is to intelligently extract product details from raw, unstructured data and format them into a clean, structured JSON format ready for downstream processing.

Your Mission:
Carefully analyze the raw input, identifying and extracting every product listed.
Format each product into a clean and consistent JSON object using the schema below.
Return a single, fully-formed JSON object only—no text, no comments, no markdown.

*Expected JSON Structure:

{
  "summary": "An interactive message to the user summarizing the results.",
  "total_count": <number of products>,
  "products": [
    {
      "product_id": "...",
      "product_name": "...",
      "price": <number>,
      "category": "...",
      "region": "...",
      "...": "any other relevant fields extracted from the raw data"
    }
  ],
  "metadata": {
    "data_source": "database | report | combined",
    "query_type": "top_products | sales_data | inventory | recommendations"
  }
}
"""

        user_prompt = f"""USER QUESTION:
{user_query}

RAW RESPONSE WITH PRODUCT DATA:
{raw_response}

ROUTE: {route}

Extract and format this as JSON following the structure specified. Output only the JSON object:"""

        try:
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Very low for consistent structured output
                max_tokens=2000
            )
            
            json_response = completion.choices[0].message.content.strip()
            
            # Try multiple ways to extract JSON
            json_str = None
            
            # Method 1: Extract JSON from markdown code blocks
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', json_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Method 2: Extract raw JSON object
                json_match = re.search(r'\{.*\}', json_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                else:
                    # Method 3: Try the entire response if it looks like JSON
                    if json_response.strip().startswith('{') and json_response.strip().endswith('}'):
                        json_str = json_response.strip()
            
            if json_str:
                try:
                    # Validate it's proper JSON
                    parsed = json.loads(json_str)
                    
                    # Return formatted JSON string
                    return json.dumps(parsed, indent=2, ensure_ascii=False)
                    
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON parsing failed for extracted string: {e}")
                    print(f"Extracted JSON string: {json_str[:200]}...")
            
            print(f"⚠️ No valid JSON found in LLM response: {json_response[:200]}...")
            return None
            
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON formatting failed: Invalid JSON generated - {e}")
            return None
        except Exception as e:
            print(f"⚠️ Product JSON formatting error: {e}")
            return None
    
    def _simple_json_fallback(self, raw_response: str, user_query: str) -> Optional[str]:
        """
        Simple fallback to convert raw response to basic JSON structure
        when LLM-based formatting fails
        """
        try:
            # Try to parse if it's already JSON-like
            if raw_response.strip().startswith('[') or raw_response.strip().startswith('{'):
                try:
                    parsed = json.loads(raw_response)
                    if isinstance(parsed, list) and len(parsed) > 0:
                        # Wrap list in basic structure
                        result = {
                            "summary": f"Found {len(parsed)} items",
                            "total_count": len(parsed),
                            "products": parsed,
                            "metadata": {
                                "data_source": "database",
                                "query_type": "product_data"
                            }
                        }
                        return json.dumps(result, indent=2, ensure_ascii=False)
                except json.JSONDecodeError:
                    pass
            
            # Try to extract from string representation of Python list/dict
            if '[{' in raw_response and '}]' in raw_response:
                # Find the list part
                import ast
                try:
                    # Use ast.literal_eval for safe evaluation
                    start = raw_response.find('[{')
                    end = raw_response.rfind('}]') + 2
                    list_str = raw_response[start:end]
                    parsed = ast.literal_eval(list_str)
                    
                    if isinstance(parsed, list) and len(parsed) > 0:
                        result = {
                            "summary": f"Found {len(parsed)} items",
                            "total_count": len(parsed),
                            "products": parsed,
                            "metadata": {
                                "data_source": "database", 
                                "query_type": "product_data"
                            }
                        }
                        return json.dumps(result, indent=2, ensure_ascii=False)
                except (ValueError, SyntaxError) as e:
                    print(f"⚠️ Simple JSON fallback failed to parse list: {e}")
                    
            return None
            
        except Exception as e:
            print(f"⚠️ Simple JSON fallback error: {e}")
            return None
    
    def _get_system_prompt(self, route: str) -> str:
        """Get route-specific system prompt"""
        
#         base_instructions = """You are a professional e-commerce assistant. Your job is to take raw data or information and present it clearly to users.

# CRITICAL RULES:
# - NEVER fabricate data, numbers, or facts not present in the raw response
# - If raw data is empty/missing, acknowledge it politely
# - Be concise but complete
# - Use natural, conversational language
# - Format lists and data for easy scanning"""

#         route_specific = {
#             'sql': """
# The raw response contains database query results with product data.
# - Present numbers and metrics clearly
# - Use bullet points for multiple items
# - Highlight key findings
# - Add brief context when helpful""",
            
#             'rag': """
# The raw response contains information from e-commerce documents/reports.
# - Synthesize information naturally
# - Cite insights without being verbose
# - Use paragraphs for concepts, bullets for lists
# - Add "According to..." or "Based on..." when appropriate""",
            
#             'orchestrator': """
# The raw response combines multiple data sources (database + documents).
# - Integrate both types of information seamlessly
# - Present a cohesive answer
# - Use sections if combining distinct insights
# - Make connections between data and context clear"""
#         }
# Updated system prompt (replace your existing strings with these)
        base_instructions = """You are a professional, friendly e-commerce assistant that converts raw product data into clear, helpful responses.

CRITICAL RULES (must follow):
- NEVER fabricate data, numbers, or facts not present in the raw response.
- If raw data is empty/missing, acknowledge it politely and offer next steps.
- Be concise but complete.
- Use natural, conversational language (warm, helpful, not robotic).
- Format lists and data for easy scanning (bullets, short lines, headings).
- Always include a short, proactive call-to-action (CTA) or follow-up question to keep the user engaged.
- Offer one brief alternative / suggestion where relevant (without inventing facts).
- If user appears to have given a specific choice in their request, echo that choice before offering alternatives."""

        route_specific = {
    'sql': """
The raw response contains database query results with product data.
- Present numbers and metrics clearly (table-like bullets).
- Use bullet points for multiple items and bold key attributes (price, stock, rating).
- Start with a one-line friendly summary (1 sentence).
- End with a CTA: ask if they want to sort/filter/see details.
- If relevant, provide one alternative suggestion (e.g., "If you'd like lower price, try: ...") but do not invent products.""",

    'rag': """
The raw response contains information from e-commerce documents/reports.
- Synthesize information naturally and concisely.
- Use paragraphs for short explanation and bullets for actionable lists.
- Preface data-driven claims with "According to..." or "Based on..." when appropriate.
- Add a short follow-up prompt: "Would you like me to...?"
- Suggest one alternative phrasing or next action the user can take.""",

    'orchestrator': """
The raw response combines multiple data sources (database + documents).
- Integrate both types of information seamlessly into sections.
- Use a concise header, a short summary sentence, then bullets for products or insights.
- Link facts to their source when useful (e.g., 'based on inventory data').
- Finish with a friendly question to guide the next step (e.g., "Want me to add these to cart?")."""
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

