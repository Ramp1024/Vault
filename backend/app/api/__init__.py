from fastapi import APIRouter

from app.api.chat import router as chat_router
from app.api.health import router as health_router

router = APIRouter()
router.include_router(health_router)
router.include_router(chat_router)