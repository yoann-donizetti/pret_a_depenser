#prod_logging.py
from __future__ import annotations
from typing import Any, Dict
from app.utils.prod_logging_db import log_event_db

def log_event(event: Dict[str, Any]) -> None:
    log_event_db(event)