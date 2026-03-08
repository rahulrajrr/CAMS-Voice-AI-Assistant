"""
services/sarvam_stt.py
-----------------------
Speech-to-Text (STT) service using Sarvam AI Saaras:v3.
Handles all three supported languages: English, Hindi, Tamil.
Auto-detects language from audio — no language input required.

Pipeline:
  1. Preprocess audio (noise reduction + VAD)
  2. Send to Sarvam AI Saaras:v3 for transcription + language detection

Why Sarvam AI?
  - Built specifically for Indian languages (22 languages supported)
  - Accurate for Tamil, Hindi, and Indian-accented English
  - Auto language detection works correctly for all three
"""

from __future__ import annotations

import logging
import httpx
from config import settings
from .audio_processor import preprocess_audio

# ── Module Logger ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"

# ── Language Code Normalisation ───────────────────────────────────────────────
SARVAM_LANG_MAP = {
    "ta-in": "ta", "ta": "ta",
    "hi-in": "hi", "hi": "hi",
    "en-in": "en", "en": "en",
}


async def transcribe_audio(
    audio_bytes: bytes,
    mime_type: str = "audio/wav",
) -> dict:
    """
    Transcribe audio using Sarvam AI Saaras:v3.
    Auto-detects language (en / hi / ta) from the audio content.

    Pipeline:
        1. Preprocess audio (noise reduction + VAD)
        2. Sarvam AI transcribes + detects language

    Args:
        audio_bytes (bytes): Raw audio bytes in any supported format.
        mime_type (str): MIME type e.g. 'audio/wav', 'audio/mp3'.

    Returns:
        dict: Result containing:
            - "transcript" (str): Transcribed text.
            - "confidence" (float): Confidence score (always 1.0 — Sarvam does not return this).
            - "detected_language" (str): Normalised language code (en / hi / ta).
            - "words" (list): Empty list — Sarvam does not return word timestamps.
            - "stt_provider" (str): Always 'sarvam'.

    Raises:
        ValueError: If audio_bytes is empty.
        RuntimeError: If Sarvam API call fails or returns no transcript.
    """
    if not audio_bytes:
        logger.error("transcribe_audio called with empty audio bytes.")
        raise ValueError("audio_bytes cannot be empty.")

    logger.info(
        f"Starting STT pipeline | "
        f"audio_size={len(audio_bytes)} bytes | mime_type={mime_type}"
    )

    try:
        # ── Step 1: Preprocess — denoise + VAD ───────────────────────────
        logger.info("Step 1: Preprocessing audio...")
        audio_bytes, mime_type = await preprocess_audio(
            audio_bytes           = audio_bytes,
            mime_type             = mime_type,
            apply_noise_reduction = False,  # Sarvam handles noise — skip for speed
            apply_vad             = False,  # Sarvam handles silence — skip for speed
        )
        logger.info(f"Preprocessing done | clean_size={len(audio_bytes)} bytes")

        # ── Step 2: Sarvam AI STT ─────────────────────────────────────────
        logger.info(f"Step 2: Sending to Sarvam AI | model={settings.sarvam_stt_model}")

        filename  = "audio.wav" if "wav" in mime_type else "audio.mp3"
        files     = {"file": (filename, audio_bytes, mime_type)}
        form_data = {
            "model":         settings.sarvam_stt_model,   # saaras:v3
            "language_code": "unknown",                    # Auto-detect language
        }
        headers = {
            "api-subscription-key": settings.sarvam_api_key,
        }

        logger.debug("Sending audio to Sarvam AI STT API...")
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                SARVAM_STT_URL,
                files   = files,
                data    = form_data,
                headers = headers,
            )

        # ── Handle API errors ─────────────────────────────────────────────
        if response.status_code != 200:
            error_body = response.text
            logger.error(
                f"Sarvam STT API error | status={response.status_code} | "
                f"body={error_body[:300]}"
            )
            raise RuntimeError(
                f"Sarvam STT API returned HTTP {response.status_code}: {error_body[:200]}"
            )

        result = response.json()
        logger.debug(f"Sarvam raw response: {str(result)[:300]}")

        # ── Parse response ────────────────────────────────────────────────
        transcript = (result.get("transcript") or "").strip()

        if not transcript:
            logger.warning("Sarvam STT returned an empty transcript.")
            raise RuntimeError(
                "No speech detected in the audio. "
                "Please check audio quality or speak closer to the microphone."
            )

        # ── Normalise detected language ───────────────────────────────────
        raw_lang          = (result.get("language_code") or "en").lower().strip()
        detected_language = SARVAM_LANG_MAP.get(raw_lang, "en")

        logger.info(
            f"STT complete | provider=sarvam | "
            f"raw_lang={raw_lang} | detected_language={detected_language} | "
            f"transcript='{transcript[:80]}'"
        )

        return {
            "transcript":        transcript,
            "confidence":        1.0,
            "detected_language": detected_language,
            "words":             [],
            "stt_provider":      "sarvam",
        }

    except (ValueError, RuntimeError):
        raise
    except httpx.TimeoutException as e:
        logger.error(f"Sarvam STT timed out: {str(e)}")
        raise RuntimeError("Sarvam STT request timed out. Please try again.") from e
    except Exception as e:
        logger.error(f"Unexpected error during Sarvam STT: {str(e)}", exc_info=True)
        raise RuntimeError(f"Audio transcription failed: {str(e)}") from e