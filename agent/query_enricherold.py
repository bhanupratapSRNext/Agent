"""
Query Context Enrichment Module

This module provides functionality to enrich vague or incomplete queries
with conversation context from rolling memory.

Example:
    Previous: "Show me menswear products"
    Current: "t-shirt"
    Enriched: "Show me t-shirt products from menswear category"
"""

from typing import List, Tuple
from openai import AsyncOpenAI


class QueryEnricher:
    """
    Enriches user queries with conversation context using LLM
    
    This class takes vague or incomplete queries and enriches them with
    context from previous conversation turns to make them more complete
    and actionable.
    """
    
    def __init__(self, api_key: str, model: str, base_url: str):
        """
        Initialize the QueryEnricher
        
        Args:
            api_key: OpenRouter API key
            model: Model to use (e.g., "openai/gpt-4o-mini")
            base_url: Base URL for API (e.g., "https://openrouter.ai/api/v1")
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
    async def enrich_query(
        self,
        query: str,
        conversation_history: List[Tuple[str, str]],
        max_query_words: int = 5
    ) -> str:
        """
        Enrich a vague query with conversation context
        
        Args:
            query: The current user query
            conversation_history: List of (role, text) tuples from conversation
            max_query_words: Maximum words in query before skipping enrichment
        
        Returns:
            Enriched query string, or original if enrichment not needed
        
        Examples:
            >>> enricher = QueryEnricher(api_key, model, base_url)
            >>> history = [("user", "Show me menswear"), ("assistant", "Here are...")]
            >>> enriched = await enricher.enrich_query("t-shirt", history)
            >>> print(enriched)
            "Show me t-shirt from menswear category"
        """
        # Skip enrichment if no history
        if not conversation_history:
            return query
        
        # Skip enrichment if query is already complete (more than max_query_words)
        if len(query.split()) > max_query_words:
            return query
        
        # Build context string from conversation history
        context_str = self._build_context_string(conversation_history)
        
        # Create enrichment prompt
        enrichment_prompt = self._create_enrichment_prompt(context_str, query)
        
        try:
            # Call LLM to enrich the query
            enriched_query = await self._call_llm_for_enrichment(enrichment_prompt)
            
            # Clean up the enriched query
            enriched_query = self._clean_enriched_query(enriched_query)
            
            # Log if enrichment occurred
            if enriched_query and enriched_query != query:
                print(f"🔄 Query enriched: '{query}' → '{enriched_query}'")
                return enriched_query
            
            return query
            
        except Exception as e:
            print(f"⚠️ Query enrichment failed: {e}")
            return query
    
    def _build_context_string(
        self,
        conversation_history: List[Tuple[str, str]],
        max_exchanges: int = 4
    ) -> str:
        """
        Build context string from conversation history
        
        Args:
            conversation_history: List of (role, text) tuples
            max_exchanges: Maximum number of exchanges to include
        
        Returns:
            Formatted context string
        """
        context_str = "Previous conversation:\n"
        
        # Take last N exchanges
        recent_history = conversation_history[-max_exchanges:]
        
        for role, text in recent_history:
            # Preview long responses (first 100 chars)
            preview = text[:100] if isinstance(text, str) else str(text)[:100]
            context_str += f"{role}: {preview}\n"
        
        return context_str
    
    def _create_enrichment_prompt(self, context_str: str, query: str) -> str:
        """
        Create the enrichment prompt for the LLM
        
        Args:
            context_str: Context from conversation history
            query: Current user query
        
        Returns:
            Formatted prompt string
        """
        prompt = f"""{context_str}

Current user query: "{query}"

Task: If the current query is vague or incomplete (like "t-shirt", "revenue", "it"), rewrite it as a complete question using context from the conversation. If the query is already complete, return it unchanged.

Rules:
1. Keep the rewritten query concise and natural
2. Include relevant context (category, product, topic) from previous conversation
3. Maintain the user's intent
4. If query is already complete, return it exactly as-is

Examples:
Previous: "Show me menswear"
Current: "t-shirt"
Rewritten: "Show me t-shirt from menswear category"

Previous: "Tell me about Electronics"
Current: "revenue"
Rewritten: "What is the revenue for Electronics category?"

Previous: "Show best products"
Current: "compare with last month"
Rewritten: "Compare best products with last month"

Previous: "What are e-commerce trends in Asia?"
Current: "what about Europe?"
Rewritten: "What are e-commerce trends in Europe?"

Respond with ONLY the rewritten query, nothing else."""
        
        return prompt
    
    async def _call_llm_for_enrichment(self, prompt: str) -> str:
        """
        Call LLM to enrich the query
        
        Args:
            prompt: The enrichment prompt
        
        Returns:
            Enriched query from LLM
        """
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,  # Low temperature for consistency
            max_tokens=100    # Enough for enriched query
        )
        
        return response.choices[0].message.content.strip()
    
    def _clean_enriched_query(self, enriched_query: str) -> str:
        """
        Clean up the enriched query
        
        Removes quotes and extra whitespace that LLM might add
        
        Args:
            enriched_query: Raw enriched query from LLM
        
        Returns:
            Cleaned query string
        """
        # Remove surrounding quotes if present
        enriched_query = enriched_query.strip('"').strip("'")
        
        # Remove extra whitespace
        enriched_query = " ".join(enriched_query.split())
        
        return enriched_query
    
    def should_enrich(
        self,
        query: str,
        has_conversation_history: bool,
        max_words: int = 5
    ) -> bool:
        """
        Determine if a query should be enriched
        
        Args:
            query: The user query
            has_conversation_history: Whether conversation history exists
            max_words: Maximum words before considering query complete
        
        Returns:
            True if query should be enriched, False otherwise
        """
        # Don't enrich if no history
        if not has_conversation_history:
            return False
        
        # Don't enrich if query is already complete
        if len(query.split()) > max_words:
            return False
        
        return True


# Convenience function for backward compatibility
async def enrich_query_with_context(
    query: str,
    conversation_history: List[Tuple[str, str]],
    api_key: str,
    model: str,
    base_url: str
) -> str:
    """
    Convenience function to enrich a query with context
    
    Args:
        query: User query to enrich
        conversation_history: List of (role, text) conversation turns
        api_key: API key for LLM
        model: Model name
        base_url: API base URL
    
    Returns:
        Enriched query string
    """
    enricher = QueryEnricher(api_key, model, base_url)
    return await enricher.enrich_query(query, conversation_history)
