"""
main.py — CAMS Voice & Chat Assistant v3.1
-------------------------------------------
Latency-optimized pipeline. Single LLM call per request.

CHAT timeline:
  Retrieval (PATH A exact):  ~0ms
  LLM (single call):         ~0.4s
  TTS + Action (parallel):   ~1.0s
  Total:                     ~1.5s ✅

VOICE timeline:
  STT:                       ~1.5s
  Retrieval (PATH A):        ~0ms
  LLM (single call):         ~0.4s
  TTS + Action (parallel):   ~1.0s
  Total:                     ~3.0s ✅

Key fixes vs previous version:
  - Removed double LLM call (was calling analyse_message twice for PATH B)
  - KB context retrieved and injected BEFORE single LLM call
  - TF-IDF removed — simple keyword search for KB is fast enough
  - Persistent HTTP client reused for all Sarvam calls
"""
from __future__ import annotations

import asyncio
import time
import uuid
import logging
from typing import Optional
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from schemas import ChatRequest, AssistantResponse
from services.groq_llm import analyse_message
from services.data_retriever import get_investor_context
from services.vector_store import get_vector_store
from action_engine import trigger_action

logging.basicConfig(
    level   = getattr(logging, settings.log_level, logging.INFO),
    format  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Persistent HTTP client ────────────────────────────────────────────────────
_http_client: Optional[httpx.AsyncClient] = None

def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0),
            limits  = httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
    return _http_client

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_http_client()
    get_vector_store()          # Preload JSON at startup — not on first request
    logger.info("Startup complete — HTTP client and vector store ready.")
    yield
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()

app = FastAPI(title="CAMS Assistant", version="3.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── STT ───────────────────────────────────────────────────────────────────────
async def _stt(audio_bytes: bytes, mime_type: str) -> dict:
    from services.sarvam_stt import SARVAM_STT_URL, SARVAM_LANG_MAP
    filename = "audio.wav" if "wav" in mime_type else "audio.webm"
    r = await get_http_client().post(
        SARVAM_STT_URL,
        files   = {"file": (filename, audio_bytes, mime_type)},
        data    = {"model": settings.sarvam_stt_model, "language_code": "unknown"},
        headers = {"api-subscription-key": settings.sarvam_api_key},
    )
    if r.status_code != 200:
        raise RuntimeError(f"STT error {r.status_code}: {r.text[:200]}")
    data       = r.json()
    transcript = (data.get("transcript") or "").strip()
    if not transcript:
        raise RuntimeError("No speech detected — please speak clearly.")
    lang = SARVAM_LANG_MAP.get((data.get("language_code") or "en").lower(), "en")
    return {"transcript": transcript, "detected_language": lang}


# ── TTS ───────────────────────────────────────────────────────────────────────
async def _tts(text: str, language: str) -> dict:
    from services.sarvam_tts import SARVAM_TTS_URL, LANGUAGE_MAP
    r = await get_http_client().post(
        SARVAM_TTS_URL,
        json = {
            "inputs":               [text[:400]],   # Cap at 400 chars — faster TTS
            "target_language_code": LANGUAGE_MAP.get(language, "en-IN"),
            "speaker":              settings.sarvam_tts_speaker,
            "model":                settings.sarvam_tts_model,
            "speech_sample_rate":   8000,           # 8kHz — 3x smaller payload
            "pace":                 1.1,
        },
        headers = {"api-subscription-key": settings.sarvam_api_key,
                   "Content-Type": "application/json"},
    )
    if r.status_code != 200:
        logger.warning(f"TTS error {r.status_code}")
        return {"audio": None}
    audios = r.json().get("audios") or []
    return {"audio": audios[0] if audios else None}


# ── Context retrieval (single path — no double LLM call) ─────────────────────
async def _get_context(
    message: str,
    investor_id: Optional[str]           = None,
    conversation_history: Optional[list] = None,
) -> tuple[Optional[dict], str]:
    """
    Resolve which investor to fetch data for, in priority order:
      1. PAN explicitly in the CURRENT message  ← always wins (user switching accounts)
      2. investor_id passed from frontend session (already verified earlier)
      3. PAN found in conversation history       (user gave it a few turns ago)
      4. Knowledge base search                   (policy/FAQ queries)

    Priority 1 > 2 ensures that when a user types a NEW PAN,
    we switch to that investor immediately instead of using the old session PAN.
    """
    import re
    PAN_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", re.IGNORECASE)
    vs     = get_vector_store()

    # ── 1. PAN in CURRENT message (highest priority) ──────────────────────
    current_pan = None
    m = PAN_RE.search(message.upper())
    if m:
        current_pan = m.group(0)
        logger.info(f"PAN in current message: {current_pan}")

    # ── 2. Session investor_id (only if no new PAN in current message) ────
    resolved_id = current_pan or investor_id or None

    # ── 3. Scan history (only if still nothing resolved) ─────────────────
    if not resolved_id and conversation_history:
        for turn in reversed(conversation_history):
            text = turn.get("content", "") or ""
            m2   = PAN_RE.search(text.upper())
            if m2:
                resolved_id = m2.group(0)
                logger.info(f"PAN from history: {resolved_id}")
                break

    # ── PATH A: exact investor lookup ─────────────────────────────────────
    if resolved_id:
        investor, account_ctx = await get_investor_context(
            user_message       = message,
            extracted_entities = {"investor_id": resolved_id},
        )
        if account_ctx:
            return investor, account_ctx

    # ── PATH B: knowledge base ────────────────────────────────────────────
    kb_ctx = await vs.search_knowledge(message, top_k=2)
    if kb_ctx:
        return None, kb_ctx

    return None, ""


# ── CHAT endpoint ─────────────────────────────────────────────────────────────
@app.post("/api/chat", response_model=AssistantResponse)
async def process_chat(request: ChatRequest) -> AssistantResponse:
    session_id = request.session_id or str(uuid.uuid4())
    t0         = time.perf_counter()

    try:
        # ── Retrieval + LLM + TTS timeline:
        # [retrieval ~0ms] → [LLM ~0.4s] → [TTS ∥ Action ~1.0s] = ~1.5s total

        t1 = time.perf_counter()
        investor, context = await _get_context(
            message              = request.message,
            investor_id          = request.investor_id,
            conversation_history = request.conversation_history or [],
        )
        logger.info(f"[CHAT] retrieval={time.perf_counter()-t1:.3f}s ctx={'account' if investor else ('kb' if context else 'none')}")

        t2       = time.perf_counter()
        analysis = await analyse_message(
            user_message         = request.message,
            language             = request.language or "en",
            investor_id          = request.investor_id or None,
            conversation_history = request.conversation_history or [],
            investor_context     = context or None,
        )
        logger.info(f"[CHAT] llm={time.perf_counter()-t2:.2f}s intent={analysis.intent}")

        t3 = time.perf_counter()
        action_result, tts_result = await asyncio.gather(
            trigger_action(analysis),
            _tts(analysis.response_text, request.language or "en"),
        )
        logger.info(f"[CHAT] tts+action={time.perf_counter()-t3:.2f}s | TOTAL={time.perf_counter()-t0:.2f}s")

        return AssistantResponse(
            session_id          = session_id,
            transcribed_text    = None,
            detected_language   = request.language or "en",
            intent              = analysis.intent,
            sentiment           = analysis.sentiment,
            entities            = analysis.entities,
            response_text       = analysis.response_text,
            confidence          = analysis.confidence,
            requires_escalation = analysis.requires_escalation,
            action_result       = action_result,
            audio_url           = tts_result.get("audio"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[CHAT] {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── VOICE endpoint ────────────────────────────────────────────────────────────
@app.post("/api/voice", response_model=AssistantResponse)
async def process_voice(
    audio_file:  UploadFile = File(...),
    session_id:  str        = Form(default=""),
    investor_id: str        = Form(default=""),
) -> AssistantResponse:
    if not session_id:
        session_id = str(uuid.uuid4())
    t0 = time.perf_counter()

    supported = {
        "audio/wav","audio/wave","audio/x-wav","audio/mpeg",
        "audio/mp3","audio/webm","audio/ogg","application/octet-stream",
    }
    if audio_file.content_type not in supported:
        raise HTTPException(400, f"Unsupported: {audio_file.content_type}")

    try:
        audio_bytes = await audio_file.read()
        if not audio_bytes:
            raise HTTPException(400, "Empty audio file.")

        # STT (unavoidable ~1.5s)
        t1  = time.perf_counter()
        stt = await _stt(audio_bytes, audio_file.content_type or "audio/wav")
        transcript, lang = stt["transcript"], stt["detected_language"]
        logger.info(f"[VOICE] stt={time.perf_counter()-t1:.2f}s lang={lang} '{transcript[:50]}'")

        # Retrieval (instant for PATH A, ~5ms for PATH B)
        t2 = time.perf_counter()
        investor, context = await _get_context(
            message              = transcript,
            investor_id          = investor_id or None,
            conversation_history = [],   # Voice history tracked via sessionInvestorId on frontend
        )
        logger.info(f"[VOICE] retrieval={time.perf_counter()-t2:.3f}s ctx={'account' if investor else ('kb' if context else 'none')}")

        # Single LLM call with full context
        t3       = time.perf_counter()
        analysis = await analyse_message(
            user_message         = transcript,
            language             = lang,
            investor_id          = investor_id or None,
            conversation_history = [],
            investor_context     = context or None,
        )
        logger.info(f"[VOICE] llm={time.perf_counter()-t3:.2f}s intent={analysis.intent}")

        # TTS + Action in parallel
        t4 = time.perf_counter()
        action_result, tts_result = await asyncio.gather(
            trigger_action(analysis),
            _tts(analysis.response_text, lang),
        )
        logger.info(f"[VOICE] tts+action={time.perf_counter()-t4:.2f}s | TOTAL={time.perf_counter()-t0:.2f}s ✅")

        return AssistantResponse(
            session_id          = session_id,
            transcribed_text    = transcript,
            detected_language   = lang,
            intent              = analysis.intent,
            sentiment           = analysis.sentiment,
            entities            = analysis.entities,
            response_text       = analysis.response_text,
            confidence          = analysis.confidence,
            requires_escalation = analysis.requires_escalation,
            action_result       = action_result,
            audio_url           = tts_result.get("audio"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[VOICE] {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host      = settings.app_host,
        port      = settings.app_port,
        reload    = False,
        log_level = "warning",
        workers   = 1,
    )