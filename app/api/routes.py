from fastapi import APIRouter, Depends, HTTPException

from app.core.logger import get_logger
from app.models.schemas import ChatRequest, ChatResponse, SourceInfo
from app.rag.generator import generate_response
from app.rag.language_detector import detect_language
from app.rag.retriever import HybridRetriever, get_retriever

logger = get_logger(__name__)
router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "ok", "service": "Plant Care RAG Chatbot"}


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, retriever: HybridRetriever = Depends(get_retriever)):
    try:
        # 1) Detect language
        lang = detect_language(request.query)
        logger.info(f"Query: '{request.query[:80]}...' | Lang: {lang}")

        # 2) Retrieve relevant chunks (cross-lingual aware)
        chunks = retriever.retrieve(request.query, lang=lang)

        if not chunks:
            fallback = (
                "I couldn't find relevant information in the plant care knowledge base. "
                "Please try rephrasing your question."
                if lang == "en"
                else "پودوں کی دیکھ بھال کی معلومات میں متعلقہ مواد نہیں ملا۔ براہ کرم اپنا سوال دوبارہ پوچھیں۔"
            )
            return ChatResponse(
                answer=fallback,
                language_detected=lang,
                sources=[],
                query=request.query,
            )

        # 3) Format chat history
        history = [{"role": m.role, "content": m.content} for m in (request.chat_history or [])]

        # 4) Generate response
        answer = generate_response(request.query, chunks, lang, history)

        # 5) Build source list (deduplicated)
        seen = set()
        sources = []
        for chunk in chunks:
            meta = chunk.get("metadata", {}) or {}
            key = (meta.get("source", ""), int(meta.get("page", 0) or 0))
            if key not in seen:
                seen.add(key)
                sources.append(SourceInfo(source=key[0], page=key[1]))

        return ChatResponse(
            answer=answer,
            language_detected=lang,
            sources=sources,
            query=request.query,
        )

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

