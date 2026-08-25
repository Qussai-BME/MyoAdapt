"""
myoadapt.api — FastAPI REST + WebSocket streaming
====================================================
"""
from myoadapt.api.rest import create_app
from myoadapt.api.websocket import create_ws_app

__all__ = ["create_app", "create_ws_app"]
