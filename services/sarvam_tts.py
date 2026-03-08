"""
services/sarvam_tts.py
-----------------------
Text-to-Speech service using Sarvam AI Bulbul v3.
Replaces Deepgram TTS for all languages — English, Hindi, and Tamil.

Why Sarvam Bulbul v3?
  - Purpose-built for Indian languages and Indian English accent
  - 30+ natural-sounding voices
  - Handles code-mixed text (e.g. Hinglish) natively
  - Supports en-IN, hi-IN, ta-IN natively
  - Returns base64-encoded WAV audio directly

API: POST https://api.sarvam.ai/text-to-speech
  - model: bulbul:v3
  - inputs: list of text strings
  - target_language_code: en-IN / hi-IN / ta-IN
  - speaker: Meera (female, clear) — works well for all 3 languages
  - Response: {"audios": ["<base64_wav>", ...]}
"""

from __future__ import annotations

import base64
import logging
import httpx
from config import settings

# ── Module Logger ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Sarvam TTS Constants ──────────────────────────────────────────────────────
SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
SARVAM_TTS_MODEL = "bulbul:v3"

# ── Language → BCP-47 code mapping ───────────────────────────────────────────
LANGUAGE_CODE_MAP: dict[str, str] = {
    "en":    "en-IN",
    "en-IN": "en-IN",
    "hi":    "hi-IN",
    "hi-IN": "hi-IN",
    "ta":    "ta-IN",
    "ta-IN": "ta-IN",
}
# Alias used by main.py
LANGUAGE_MAP = LANGUAGE_CODE_MAP

# ── Language → Speaker mapping ────────────────────────────────────────────────
LANGUAGE_SPEAKER_MAP: dict[str, str] = {
    "en-IN": "anushka",    # Clear female English (Indian accent)
    "hi-IN": "anushka",    # Natural Hindi female voice
    "ta-IN": "anushka",    # Natural Tamil female voice
}
DEFAULT_SPEAKER = "anushka"


async def synthesize_speech(
    text: str,
    detected_language: str = "en",
    encode_base64: bool = True,
) -> dict:
    """
    Convert text to speech using Sarvam AI Bulbul v3.
    Responds in the same language as the detected input language.

    Args:
        text (str): Text to synthesize. Max 2500 characters (Bulbul v3 limit).
        detected_language (str): Short language code from STT e.g. 'en', 'hi', 'ta'.
                                  Defaults to 'en'.
        encode_base64 (bool): If True returns base64 string (for JSON transport).
                              If False returns raw bytes. Defaults to True.

    Returns:
        dict: Result containing:
            - "audio" (str | bytes): Base64 string or raw WAV bytes.
            - "voice_model" (str): TTS model used.
            - "language" (str): BCP-47 language code used.
            - "speaker" (str): Speaker voice used.
            - "character_count" (int): Number of characters synthesized.

    Raises:
        ValueError: If text is empty.
        RuntimeError: If Sarvam TTS API call fails.
    """
    if not text or not text.strip():
        logger.error("synthesize_speech called with empty text.")
        raise ValueError("Text for TTS cannot be empty.")

    text = text.strip()

    # Bulbul v3 supports up to 2500 characters
    if len(text) > 2500:
        logger.warning(f"TTS text too long ({len(text)} chars). Truncating to 2500.")
        text = text[:2500]

    # ── Resolve language code and speaker ─────────────────────────────────
    language_code = LANGUAGE_CODE_MAP.get(detected_language, "en-IN")
    speaker       = LANGUAGE_SPEAKER_MAP.get(language_code, DEFAULT_SPEAKER)

    logger.info(
        f"Starting Sarvam TTS | detected_language={detected_language} | "
        f"language_code={language_code} | speaker={speaker} | "
        f"char_count={len(text)}"
    )

    try:
        payload = {
            "inputs":               [text],
            "target_language_code": language_code,
            "speaker":              settings.sarvam_tts_speaker,
            "model":                settings.sarvam_tts_model,
            "speech_sample_rate":   24000,
            "pace":                 1.0,
            "temperature":          0.6,
        }
        headers = {
            "api-subscription-key": settings.sarvam_api_key,
            "Content-Type":         "application/json",
        }

        logger.debug("Sending text to Sarvam AI Bulbul v3 TTS...")

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                SARVAM_TTS_URL,
                json    = payload,
                headers = headers,
            )

        # ── Handle API errors ─────────────────────────────────────────────
        if response.status_code != 200:
            error_body = response.text
            logger.error(
                f"Sarvam TTS API error | status={response.status_code} | "
                f"body={error_body[:300]}"
            )
            raise RuntimeError(
                f"Sarvam TTS API returned HTTP {response.status_code}: {error_body[:200]}"
            )

        result = response.json()
        logger.debug(f"Sarvam TTS raw response keys: {list(result.keys())}")

        # ── Extract base64 audio from response ────────────────────────────
        # Sarvam returns: {"audios": ["<base64_wav_string>", ...]}
        audios = result.get("audios", [])
        if not audios or not audios[0]:
            logger.error("Sarvam TTS returned empty audios list.")
            raise RuntimeError("Sarvam TTS returned empty audio. Please retry.")

        audio_b64: str = audios[0]   # Already base64 encoded by Sarvam

        # ── Decode to raw bytes for size logging ──────────────────────────
        audio_bytes = base64.b64decode(audio_b64)

        logger.info(
            f"Sarvam TTS successful | audio_size={len(audio_bytes)} bytes | "
            f"language={language_code} | speaker={speaker}"
        )

        # Return base64 or raw bytes based on caller preference
        audio_output = audio_b64 if encode_base64 else audio_bytes

        return {
            "audio":           audio_output,
            "voice_model":     SARVAM_TTS_MODEL,
            "language":        language_code,
            "speaker":         speaker,
            "character_count": len(text),
        }

    except (ValueError, RuntimeError):
        raise
    except httpx.TimeoutException as e:
        logger.error(f"Sarvam TTS request timed out: {str(e)}")
        raise RuntimeError("Sarvam TTS request timed out. Please try again.") from e
    except Exception as e:
        logger.error(
            f"Unexpected error during Sarvam TTS: {str(e)}", exc_info=True
        )
        raise RuntimeError(f"Speech synthesis failed: {str(e)}") from e