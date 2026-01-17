from fastapi import APIRouter

from src.routes.auth import router as auth_router
from src.routes.routes import router as chat_router

router = APIRouter()
router.include_router(chat_router)
router.include_router(auth_router)

__all__ = ["router"]
