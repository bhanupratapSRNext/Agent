from langgraph.checkpoint.memory import MemorySaver

def make_checkpointer():
    # Swap to Redis/DB-backed later for persistence.
    return MemorySaver()
