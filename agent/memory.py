# from langgraph.checkpoint.memory import MemorySaver

# def make_checkpointer():
#     # Swap to Redis/DB-backed later for persistence.
#     return MemorySaver()


from collections import defaultdict, deque
from typing import Deque, Dict, List, Tuple

class RollingMemory:
    def __init__(self, window_size: int = 6):
        self.window_size = window_size
        self._store: Dict[str, Deque[Tuple[str, str]]] = defaultdict(lambda: deque(maxlen=self.window_size))

    def append(self, session_id: str, role: str, text: str):
        self._store[session_id].append((role, text))

    def get(self, session_id: str) -> List[Tuple[str, str]]:
        return list(self._store[session_id])

    def clear(self, session_id: str):
        self._store.pop(session_id, None)
