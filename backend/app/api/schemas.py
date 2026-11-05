"""Pydantic request/response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str = "0.1.0"


class SettingsUpdate(BaseModel):
    model_config = {"extra": "forbid"}

    performance_profile: Optional[str] = None
    language_mode: Optional[str] = None
    auto_language_routing: Optional[bool] = None
    primary_language: Optional[str] = None
    secondary_language: Optional[str] = None
    stt_provider: Optional[str] = None
    stt_model: Optional[str] = None
    moonshine_model: Optional[str] = None
    tts_voice: Optional[str] = None
    tts_speed: Optional[float] = None
    bengali_tts_provider: Optional[str] = None
    bengali_tts_voice: Optional[str] = None
    english_tts_provider: Optional[str] = None
    ollama_model: Optional[str] = None
    ollama_model_fallback: Optional[str] = None
    vad_confidence: Optional[float] = None
    vad_start_secs: Optional[float] = None
    vad_stop_secs: Optional[float] = None
    smart_turn_stop_secs: Optional[float] = None
    max_response_sentences: Optional[int] = None
    llm_temperature: Optional[float] = None
    llm_context_tokens: Optional[int] = None
    thinking_mode: Optional[bool] = None
    system_prompt: Optional[str] = None
    memory_consent_default: Optional[bool] = None
    memory_top_k: Optional[int] = None
    embedding_model: Optional[str] = None
    push_to_talk: Optional[bool] = None
    tts_chunk_min_chars: Optional[int] = None
    tts_chunk_max_chars: Optional[int] = None


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    language: Optional[str] = None  # bangla | english | auto
    voice: Optional[str] = None
    generation_id: Optional[str] = None


class SessionCreate(BaseModel):
    title: str = "New conversation"


class MemoryCreate(BaseModel):
    content: str = Field(min_length=1, max_length=4000)
    memory_type: str = "fact"
    importance: float = Field(default=0.7, ge=0, le=1)
    confidence: float = Field(default=0.9, ge=0, le=1)


class MemoryUpdate(BaseModel):
    content: Optional[str] = Field(default=None, min_length=1, max_length=4000)
    memory_type: Optional[str] = None
    importance: Optional[float] = Field(default=None, ge=0, le=1)
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    active: Optional[bool] = None


class ConsentUpdate(BaseModel):
    consent: bool


class NoteCreate(BaseModel):
    title: str = "Untitled"
    content: str = ""
    tags: list[str] = Field(default_factory=list)


class TaskCreate(BaseModel):
    title: str = Field(min_length=1, max_length=256)
    description: str = ""


class TaskUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None


class ChatRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)
    session_id: Optional[str] = None


class ToolRequest(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    confirmed: bool = False
    session_id: Optional[str] = None


class SpokenAck(BaseModel):
    message_id: str
    status: str = "spoken"
    generation_id: Optional[str] = None


class InterruptRequest(BaseModel):
    session_id: Optional[str] = None
