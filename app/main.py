import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Starting Plant Care RAG Chatbot...")

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import get_settings
from app.core.logger import get_logger
from app.rag.retriever import get_retriever

settings = get_settings()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load retriever on startup so first request isn't slow
    logger.info("Pre-loading retriever on startup...")
    get_retriever()
    logger.info("Retriever ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    description="Bilingual (English/Urdu) Plant Care RAG Chatbot API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "message": "Plant Care RAG Chatbot API",
        "docs": "/docs",
        "health": "/api/v1/health",
    }

