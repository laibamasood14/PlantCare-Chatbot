"""
Response Generator with RAG Enhancement Techniques:

1. LANGUAGE-AWARE PROMPTING  — Different system prompts for English vs Urdu
2. GROUNDING GUARD           — System prompt instructs model to stay within retrieved context
3. SOURCE CITATION           — Response includes which PDF sources were used
"""

from typing import Any, Dict, List, Optional

from groq import Groq

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


SYSTEM_PROMPT_EN = """You are a knowledgeable and friendly Plant Care Assistant.

You ONLY answer questions related to plant care, gardening, horticulture, and botany.
If the user asks something unrelated to plants, politely redirect them to ask about plant care.

Use ONLY the provided context to answer. If the context does not contain enough information, say so honestly — do not fabricate.
Reason through the context step by step before giving your final answer.
When relevant, mention which source (PDF name) supports your answer.
Keep answers clear, practical, and actionable.
Respond in ENGLISH."""


SYSTEM_PROMPT_UR = """آپ ایک ماہر اور دوستانہ پودوں کی دیکھ بھال کے معاون ہیں۔

آپ صرف پودوں کی دیکھ بھال، باغبانی، اور نباتیات سے متعلق سوالات کے جوابات دیتے ہیں۔
اگر صارف کوئی غیر متعلق سوال پوچھے تو انہیں پودوں سے متعلق سوال پوچھنے کی ترغیب دیں۔

صرف فراہم کردہ سیاق و سباق کو استعمال کرتے ہوئے جواب دیں۔ اگر سیاق و سباق میں کافی معلومات نہ ہوں تو صادقانہ طور پر کہیں — کچھ بھی بناوٹی نہ کہیں۔
جواب دینے سے پہلے سیاق و سباق کا قدم بہ قدم جائزہ لیں۔
جب ممکن ہو تو بتائیں کہ کس ذریعہ (PDF نام) سے معلومات لی گئی ہیں۔
جوابات واضح، عملی اور قابل عمل ہونے چاہیے۔
اردو میں جواب دیں۔"""


def _get_client() -> Groq:
    if not settings.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env or environment variables.")
    return Groq(api_key=settings.GROQ_API_KEY)


def build_context_block(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a numbered context block."""
    parts: List[str] = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {}) or {}
        source = meta.get("source", "unknown")
        page = meta.get("page", "?")
        parts.append(f"[{i}] (Source: {source}, Page: {page})\n{chunk.get('text', '')}")
    return "\n\n---\n\n".join(parts)


def generate_response(
    query: str,
    chunks: List[Dict[str, Any]],
    lang: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Generate a grounded, language-appropriate plant care response using Groq.
    chat_history: list of {"role": "user"/"assistant", "content": "..."} for multi-turn context.
    """
    system_prompt = SYSTEM_PROMPT_UR if lang == "ur" else SYSTEM_PROMPT_EN
    context_block = build_context_block(chunks)

    context_message = (
        f"Retrieved Plant Care Context:\n\n{context_block}\n\n"
        f"---\nNow answer the following query using the above context."
    )

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    # Include prior conversation turns (last 6 turns max)
    if chat_history:
        messages.extend(chat_history[-6:])

    messages.append({"role": "user", "content": f"{context_message}\n\nQuery: {query}"})

    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=messages,
            temperature=settings.TEMPERATURE,
            max_tokens=1024,
        )
        answer = response.choices[0].message.content.strip()
        usage = getattr(response, "usage", None)
        if usage and getattr(usage, "total_tokens", None) is not None:
            logger.info(f"Generated response (lang={lang}, tokens={usage.total_tokens})")
        else:
            logger.info(f"Generated response (lang={lang})")
        return answer
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        raise