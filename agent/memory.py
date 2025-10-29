# from langgraph.checkpoint.memory import MemorySaver

# def make_checkpointer():
#     # Swap to Redis/DB-backed later for persistence.
#     return MemorySaver()


from collections import defaultdict, deque
from typing import Deque, Dict, List, Tuple, Optional
import hashlib
from datetime import datetime, timedelta

class RollingMemory:
    def __init__(self, window_size: int = 10, cache_ttl_seconds: int = 3600, max_cache_size: int = 200):
        """
        Enhanced memory that handles both conversation history and response caching
        
        Args:
            window_size: Number of conversation turns to keep per session
            cache_ttl_seconds: How long to keep cached responses (in seconds)
            max_cache_size: Maximum number of cached responses to store
        """
        self.window_size = window_size
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self.max_cache_size = max_cache_size
        
        # Conversation history: session_id -> deque of (role, text) tuples
        self._store: Dict[str, Deque[Tuple[str, str]]] = defaultdict(lambda: deque(maxlen=self.window_size))
        
        # Response cache: (session_id, query_hash) -> (response, timestamp)
        self._response_cache: Dict[Tuple[str, str], Tuple[str, datetime]] = {}
        
        # Cache statistics
        self.cache_hits = 0
        self.cache_misses = 0

    def append(self, session_id: str, role: str, text: str):
        """Add a conversation turn to memory"""
        self._store[session_id].append((role, text))

    def get(self, session_id: str) -> List[Tuple[str, str]]:
        """Get conversation history for a session"""
        return list(self._store[session_id])

    def clear(self, session_id: str):
        """Clear conversation history for a session"""
        self._store.pop(session_id, None)
    
    def clear_session_cache(self, session_id: str):
        """Clear cached responses for a specific session"""
        keys_to_remove = [key for key in self._response_cache.keys() if key[0] == session_id]
        for key in keys_to_remove:
            del self._response_cache[key]
        
    def clear_all_cache(self):
        """Clear all cached responses"""
        self._response_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _hash_query(self, query: str) -> str:
        """Create a hash for the query to use as cache key"""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def get_cached_response(self, query: str, session_id: str = "default") -> Optional[str]:
        """Get cached response if available and not expired"""
        query_hash = self._hash_query(query)
        cache_key = (session_id, query_hash)
        
        if cache_key in self._response_cache:
            response, timestamp = self._response_cache[cache_key]
            
            # Check if expired
            if datetime.now() - timestamp < self.cache_ttl:
                self.cache_hits += 1
                return response
            else:
                # Remove expired entry
                del self._response_cache[cache_key]
        
        self.cache_misses += 1
        return None
    
    def cache_response(self, query: str, response: str, session_id: str = "default"):
        """Cache a response"""
        query_hash = self._hash_query(query)
        cache_key = (session_id, query_hash)
        
        # Evict oldest if at max size
        if len(self._response_cache) >= self.max_cache_size:
            oldest_key = min(self._response_cache.keys(), 
                           key=lambda k: self._response_cache[k][1])
            del self._response_cache[oldest_key]
        
        self._response_cache[cache_key] = (response, datetime.now())
    
    def get_cache_stats(self) -> Dict[str, any]:
        """Get cache statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        # Count cache entries per session
        sessions_with_cache = {}
        for (session_id, query_hash), (response, timestamp) in self._response_cache.items():
            sessions_with_cache[session_id] = sessions_with_cache.get(session_id, 0) + 1
        
        return {
            'size': len(self._response_cache),
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_rate': f"{hit_rate:.1f}%",
            'conversation_sessions': len(self._store),
            'cached_sessions': len(sessions_with_cache),
            'cache_per_session': sessions_with_cache
        }
