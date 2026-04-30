# db/__init__.py - Database package
from .models import Database, Person, Embedding, Event, Camera, Detection
from .connection import get_db, init_db, close_db

__all__ = ["Database", "Person", "Embedding", "Event", "Camera", "Detection", "get_db", "init_db", "close_db"]
