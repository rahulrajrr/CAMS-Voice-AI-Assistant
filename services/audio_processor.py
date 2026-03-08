"""
services/audio_processor.py
----------------------------
Audio preprocessing pipeline for the Voice & Chat Assistant.
Runs before Deepgram STT to improve transcription accuracy on:
  - Noisy audio (background noise, mic hiss, static)
  - Low clarity recordings
  - Phone/microphone quality audio

Pipeline:
  1. Format Conversion  → Convert any format (MP3, OGG, WebM) to 16kHz mono WAV
  2. Noise Reduction    → Spectral subtraction via noisereduce
  3. VAD (Voice Activity Detection) → Strip silent/non-speech segments via Silero VAD
  4. Return clean WAV bytes ready for Deepgram STT
"""

from __future__ import annotations

import io
import logging

import numpy as np
import noisereduce as nr
import torch
import torchaudio

from silero_vad import load_silero_vad, get_speech_timestamps
from pydub import AudioSegment

# ── Module Logger ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_SAMPLE_RATE   = 16000
TARGET_CHANNELS      = 1
MIN_SPEECH_DURATION  = 500


# ── Load Silero VAD Model (loaded once globally) ─────────────────────────────
VAD_MODEL = load_silero_vad()


def _convert_to_wav(audio_bytes: bytes, mime_type: str) -> AudioSegment:
    try:
        fmt_map = {
            "audio/mpeg": "mp3",
            "audio/mp3": "mp3",
            "audio/wav": "wav",
            "audio/wave": "wav",
            "audio/x-wav": "wav",
            "audio/ogg": "ogg",
            "audio/webm": "webm",
        }

        fmt = fmt_map.get(mime_type, "wav")

        segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
        segment = segment.set_frame_rate(TARGET_SAMPLE_RATE)
        segment = segment.set_channels(TARGET_CHANNELS)
        segment = segment.set_sample_width(2)

        logger.debug(
            f"Audio converted | format={fmt} | duration={len(segment)}ms | "
            f"sample_rate={segment.frame_rate}Hz | channels={segment.channels}"
        )

        return segment

    except Exception as e:
        logger.error(f"Audio format conversion failed: {str(e)}")
        raise RuntimeError(f"Failed to convert audio format: {str(e)}") from e


def _reduce_noise(segment: AudioSegment) -> AudioSegment:

    try:

        samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
        samples = samples / 32768.0

        reduced = nr.reduce_noise(
            y=samples,
            sr=TARGET_SAMPLE_RATE,
            stationary=True,
            prop_decrease=0.75,
        )

        reduced_int16 = (reduced * 32768.0).astype(np.int16)

        clean_segment = AudioSegment(
            data=reduced_int16.tobytes(),
            sample_width=2,
            frame_rate=TARGET_SAMPLE_RATE,
            channels=TARGET_CHANNELS,
        )

        logger.debug(
            f"Noise reduction applied | "
            f"original_rms={segment.rms} | clean_rms={clean_segment.rms}"
        )

        return clean_segment

    except Exception as e:
        logger.warning(
            f"Noise reduction failed — using original audio. Error: {str(e)}"
        )
        return segment


def _apply_vad(segment: AudioSegment) -> AudioSegment:
    """
    Apply Voice Activity Detection using Silero VAD.
    """

    try:

        samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
        samples = samples / 32768.0

        audio_tensor = torch.from_numpy(samples)

        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            VAD_MODEL,
            sampling_rate=TARGET_SAMPLE_RATE,
        )

        if not speech_timestamps:
            logger.warning("VAD: No speech detected — returning original audio.")
            return segment

        speech_segments = []

        for ts in speech_timestamps:

            start = int(ts["start"])
            end = int(ts["end"])

            start_byte = start * 2
            end_byte = end * 2

            speech_segments.append(segment.raw_data[start_byte:end_byte])

        speech_bytes = b"".join(speech_segments)

        speech_segment = AudioSegment(
            data=speech_bytes,
            sample_width=2,
            frame_rate=TARGET_SAMPLE_RATE,
            channels=TARGET_CHANNELS,
        )

        if len(speech_segment) < MIN_SPEECH_DURATION:
            logger.warning(
                f"VAD result too short ({len(speech_segment)}ms < {MIN_SPEECH_DURATION}ms). "
                f"Returning original audio."
            )
            return segment

        logger.debug(
            f"VAD complete | original={len(segment)}ms | "
            f"after_vad={len(speech_segment)}ms | "
            f"removed={len(segment) - len(speech_segment)}ms of silence"
        )

        return speech_segment

    except Exception as e:
        logger.warning(
            f"VAD processing failed — using original audio. Error: {str(e)}"
        )
        return segment


def _segment_to_wav_bytes(segment: AudioSegment) -> bytes:

    try:

        buffer = io.BytesIO()
        segment.export(buffer, format="wav")
        wav_bytes = buffer.getvalue()

        logger.debug(f"Exported WAV | size={len(wav_bytes)} bytes")

        return wav_bytes

    except Exception as e:
        logger.error(f"WAV export failed: {str(e)}")
        raise RuntimeError(f"Failed to export audio as WAV: {str(e)}") from e


async def preprocess_audio(
    audio_bytes: bytes,
    mime_type: str = "audio/wav",
    apply_noise_reduction: bool = True,
    apply_vad: bool = True,
) -> tuple[bytes, str]:

    if not audio_bytes:
        raise ValueError("audio_bytes cannot be empty.")

    logger.info(
        f"Starting audio preprocessing | mime={mime_type} | "
        f"size={len(audio_bytes)} bytes | "
        f"noise_reduction={apply_noise_reduction} | vad={apply_vad}"
    )

    try:

        segment = _convert_to_wav(audio_bytes, mime_type)

        logger.info(
            f"Step 1 complete — format converted | duration={len(segment)}ms"
        )

        if apply_noise_reduction:

            segment = _reduce_noise(segment)

            logger.info(
                f"Step 2 complete — noise reduced | duration={len(segment)}ms"
            )

        else:

            logger.info("Step 2 skipped — noise reduction disabled")

        if apply_vad:

            segment = _apply_vad(segment)

            logger.info(
                f"Step 3 complete — VAD applied | duration={len(segment)}ms"
            )

        else:

            logger.info("Step 3 skipped — VAD disabled")

        clean_bytes = _segment_to_wav_bytes(segment)

        logger.info(
            f"Audio preprocessing complete | "
            f"input_size={len(audio_bytes)} bytes | "
            f"output_size={len(clean_bytes)} bytes | "
            f"duration={len(segment)}ms"
        )

        return clean_bytes, "audio/wav"

    except ValueError:
        raise

    except RuntimeError:
        raise

    except Exception as e:

        logger.error(
            f"Unexpected error in audio preprocessing: {str(e)}",
            exc_info=True,
        )

        logger.warning(
            "Falling back to original audio due to preprocessing error."
        )

        return audio_bytes, mime_type