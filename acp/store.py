import time
from typing import Dict, List, Any
from .models import RunStatus

RUNS: Dict[str, RunStatus] = {}
EVENTS: Dict[str, List[Dict[str, Any]]] = {}

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
