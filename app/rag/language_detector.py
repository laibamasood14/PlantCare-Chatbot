"""
Language detection using:
1. lingua-language-detector (primary — high accuracy, handles short Urdu text)
2. langdetect (fallback)
3. Script-based heuristic: if text contains Unicode Urdu/Arabic script characters → Urdu
"""

import re

from langdetect import DetectorFactory, detect
from lingua import Language, LanguageDetectorBuilder

DetectorFactory.seed = 42

# Build detector for English and Urdu only (faster, more accurate scoped detectors)
_detector = (
    LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.URDU)
    .with_preloaded_language_models()
    .build()
)

URDU_SCRIPT_PATTERN = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")


def detect_language(text: str) -> str:
    """
    Returns 'ur' for Urdu, 'en' for English.
    Defaults to 'en' if detection is uncertain.
    """
    text = (text or "").strip()
    if not text:
        return "en"

    # Heuristic: if >20% characters are Urdu/Arabic script → Urdu
    urdu_chars = len(URDU_SCRIPT_PATTERN.findall(text))
    if urdu_chars / max(len(text), 1) > 0.2:
        return "ur"

    # lingua primary detection
    try:
        lang = _detector.detect_language_of(text)
        if lang == Language.URDU:
            return "ur"
        if lang == Language.ENGLISH:
            return "en"
    except Exception:
        pass

    # langdetect fallback
    try:
        code = detect(text)
        if code in ("ur", "ar"):  # langdetect sometimes codes Urdu as 'ar'
            return "ur"
    except Exception:
        pass

    return "en"  # safe default

