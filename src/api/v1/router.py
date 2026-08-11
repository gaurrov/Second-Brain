"""
Aggregates all v1 endpoint routers into a single APIRouter that main.py
mounts under the configured API prefix. Adding a new resource module
(e.g. documents.py, chat.py) means creating the endpoint file and
including its router here — nothing else needs to change.
"""
from fastapi import APIRouter

from src.api.v1.endpoints import auth, chat, documents, users

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(users.router)
api_router.include_router(documents.router)
api_router.include_router(chat.router)
