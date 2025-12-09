#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HOLOTHÉIA VOICE API — Routes vocales avec ElevenLabs

Endpoints:
- POST /api/voice/transcribe — Transcription audio (Whisper/Deepgram)
- POST /api/voice/synthesize — Synthèse vocale (ElevenLabs)
- POST /api/voice/synthesize-stream — Synthèse streaming
- GET /api/voice/config — Configuration vocale

Date: 2025-12-09
"""

import os
import io
import logging
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Header
from pydantic import BaseModel, Field
import httpx

logger = logging.getLogger("holotheia.voice")

# ============================================================================
# CONFIGURATION
# ============================================================================

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel voice
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ElevenLabs settings
ELEVENLABS_MODEL = "eleven_multilingual_v2"  # Support français + émotions
ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1"

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    voice_id: Optional[str] = Field(None, description="ElevenLabs voice ID")
    stability: float = Field(0.5, ge=0.0, le=1.0, description="Voice stability")
    similarity_boost: float = Field(0.75, ge=0.0, le=1.0, description="Similarity boost")
    style: float = Field(0.0, ge=0.0, le=1.0, description="Style exaggeration")
    use_speaker_boost: bool = Field(True, description="Use speaker boost")


class TranscriptionResponse(BaseModel):
    text: str
    language: Optional[str] = None
    duration: Optional[float] = None


class VoiceConfig(BaseModel):
    available_voices: list
    current_voice_id: str
    model: str
    features: dict


# ============================================================================
# ROUTER
# ============================================================================

router = APIRouter(prefix="/api/voice", tags=["voice"])


# ============================================================================
# TRANSCRIPTION (Whisper via OpenAI)
# ============================================================================

@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    audio: UploadFile = File(...),
    authorization: Optional[str] = Header(None)
):
    """
    Transcrit audio en texte via OpenAI Whisper

    Formats supportés: mp3, mp4, mpeg, mpga, m4a, wav, webm
    """

    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY not configured"
        )

    try:
        # Lire le fichier audio
        audio_data = await audio.read()

        if len(audio_data) < 100:
            raise HTTPException(
                status_code=400,
                detail="Audio file too small or empty"
            )

        logger.info(f"Transcribing audio: {len(audio_data)} bytes, type: {audio.content_type}")

        # Appel API OpenAI Whisper
        async with httpx.AsyncClient(timeout=30.0) as client:
            files = {
                'file': (audio.filename or 'audio.webm', audio_data, audio.content_type or 'audio/webm')
            }
            data = {
                'model': 'whisper-1',
                'language': 'fr',  # Force français pour meilleure précision
                'response_format': 'json'
            }
            headers = {
                'Authorization': f'Bearer {OPENAI_API_KEY}'
            }

            response = await client.post(
                'https://api.openai.com/v1/audio/transcriptions',
                files=files,
                data=data,
                headers=headers
            )

            if response.status_code != 200:
                error_detail = response.text
                logger.error(f"Whisper API error: {response.status_code} - {error_detail}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Transcription failed: {error_detail}"
                )

            result = response.json()
            transcription_text = result.get('text', '').strip()

            logger.info(f"Transcription success: {transcription_text[:100]}...")

            return TranscriptionResponse(
                text=transcription_text,
                language=result.get('language'),
                duration=result.get('duration')
            )

    except httpx.HTTPError as e:
        logger.error(f"HTTP error during transcription: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Network error: {str(e)}"
        )

    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )


# ============================================================================
# SYNTHÈSE VOCALE (ElevenLabs)
# ============================================================================

@router.post("/synthesize")
async def synthesize_speech(
    request: SynthesizeRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Synthétise texte en audio via ElevenLabs

    Returns: Audio stream (MP3)
    """

    if not ELEVENLABS_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="ELEVENLABS_API_KEY not configured"
        )

    try:
        voice_id = request.voice_id or ELEVENLABS_VOICE_ID

        logger.info(f"Synthesizing text: {request.text[:100]}... with voice: {voice_id}")

        # Configuration voix
        voice_settings = {
            "stability": request.stability,
            "similarity_boost": request.similarity_boost,
            "style": request.style,
            "use_speaker_boost": request.use_speaker_boost
        }

        # Payload ElevenLabs
        payload = {
            "text": request.text,
            "model_id": ELEVENLABS_MODEL,
            "voice_settings": voice_settings
        }

        # Headers
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY
        }

        # Appel API ElevenLabs
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{ELEVENLABS_API_URL}/text-to-speech/{voice_id}",
                json=payload,
                headers=headers
            )

            if response.status_code != 200:
                error_detail = response.text
                logger.error(f"ElevenLabs API error: {response.status_code} - {error_detail}")

                # Handle quota errors
                if response.status_code == 401:
                    raise HTTPException(
                        status_code=401,
                        detail="Invalid ElevenLabs API key"
                    )
                elif response.status_code == 403:
                    raise HTTPException(
                        status_code=403,
                        detail="ElevenLabs quota exceeded or subscription required"
                    )

                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Speech synthesis failed: {error_detail}"
                )

            audio_data = response.content

            logger.info(f"Synthesis success: {len(audio_data)} bytes")

            # Return audio as binary response
            from fastapi.responses import Response
            return Response(
                content=audio_data,
                media_type="audio/mpeg",
                headers={
                    "Content-Disposition": "attachment; filename=speech.mp3"
                }
            )

    except httpx.HTTPError as e:
        logger.error(f"HTTP error during synthesis: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Network error: {str(e)}"
        )

    except Exception as e:
        logger.error(f"Synthesis error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Speech synthesis failed: {str(e)}"
        )


# ============================================================================
# SYNTHÈSE STREAMING (pour réponses longues)
# ============================================================================

@router.post("/synthesize-stream")
async def synthesize_speech_stream(
    request: SynthesizeRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Synthétise texte en audio avec streaming
    Utilisé pour réponses longues en temps réel
    """

    if not ELEVENLABS_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="ELEVENLABS_API_KEY not configured"
        )

    try:
        voice_id = request.voice_id or ELEVENLABS_VOICE_ID

        logger.info(f"Streaming synthesis: {request.text[:100]}...")

        voice_settings = {
            "stability": request.stability,
            "similarity_boost": request.similarity_boost,
            "style": request.style,
            "use_speaker_boost": request.use_speaker_boost
        }

        payload = {
            "text": request.text,
            "model_id": ELEVENLABS_MODEL,
            "voice_settings": voice_settings
        }

        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY
        }

        # Streaming response
        from fastapi.responses import StreamingResponse

        async def audio_stream():
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{ELEVENLABS_API_URL}/text-to-speech/{voice_id}/stream",
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status_code != 200:
                        error = await response.aread()
                        logger.error(f"Stream error: {error.decode()}")
                        raise HTTPException(
                            status_code=response.status_code,
                            detail=f"Streaming failed: {error.decode()}"
                        )

                    async for chunk in response.aiter_bytes(chunk_size=4096):
                        yield chunk

        return StreamingResponse(
            audio_stream(),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "attachment; filename=speech.mp3"
            }
        )

    except Exception as e:
        logger.error(f"Streaming synthesis error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Streaming synthesis failed: {str(e)}"
        )


# ============================================================================
# CONFIGURATION VOCALE
# ============================================================================

@router.get("/config", response_model=VoiceConfig)
async def get_voice_config():
    """Retourne configuration vocale disponible"""

    # Voix ElevenLabs disponibles (pre-configured)
    available_voices = [
        {
            "id": "21m00Tcm4TlvDq8ikWAM",
            "name": "Rachel",
            "gender": "female",
            "language": "en",
            "description": "Calm, warm, professional"
        },
        {
            "id": "AZnzlk1XvdvUeBnXmlld",
            "name": "Domi",
            "gender": "female",
            "language": "en",
            "description": "Energetic, enthusiastic"
        },
        {
            "id": "EXAVITQu4vr4xnSDxMaL",
            "name": "Bella",
            "gender": "female",
            "language": "en",
            "description": "Soft, gentle, soothing"
        },
        {
            "id": "ErXwobaYiN019PkySvjV",
            "name": "Antoni",
            "gender": "male",
            "language": "en",
            "description": "Deep, authoritative"
        },
        {
            "id": "VR6AewLTigWG4xSOukaG",
            "name": "Arnold",
            "gender": "male",
            "language": "en",
            "description": "Strong, confident"
        }
    ]

    features = {
        "multilingual": True,
        "streaming": True,
        "voice_cloning": False,
        "emotion_control": True,
        "max_text_length": 5000,
        "supported_formats": ["mp3"],
        "supported_languages": ["en", "fr", "es", "de", "it", "pt", "pl", "hi", "zh"]
    }

    return VoiceConfig(
        available_voices=available_voices,
        current_voice_id=ELEVENLABS_VOICE_ID,
        model=ELEVENLABS_MODEL,
        features=features
    )


# ============================================================================
# HEALTH CHECK
# ============================================================================

@router.get("/health")
async def voice_health_check():
    """Health check pour services vocaux"""

    status = {
        "elevenlabs": ELEVENLABS_API_KEY != "",
        "openai_whisper": OPENAI_API_KEY != "",
        "voice_id": ELEVENLABS_VOICE_ID,
        "model": ELEVENLABS_MODEL
    }

    return {
        "status": "healthy" if all(status.values()) else "degraded",
        "services": status
    }
