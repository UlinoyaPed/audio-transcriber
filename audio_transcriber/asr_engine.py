"""Small, backend-neutral ASR interface used by MiMo orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

from .mimo_api import MimoApiRecognizer


@dataclass(frozen=True)
class ASRResult:
    """Normalized result returned by an ASR engine."""

    text: str
    language: Optional[str] = None
    confidence: Optional[float] = None


class ASREngine(Protocol):
    """Recognize one audio segment without owning retry policy."""

    def transcribe(self, audio_path: str, audio_tag: str) -> ASRResult:
        """Return a normalized recognition result."""


class MimoApiEngine:
    """Adapt :class:`MimoApiRecognizer` to the backend-neutral interface."""

    def __init__(self, recognizer: MimoApiRecognizer) -> None:
        self.recognizer = recognizer

    def transcribe(self, audio_path: str, audio_tag: str) -> ASRResult:
        return ASRResult(
            text=self.recognizer.transcribe(audio_path, audio_tag),
            language=audio_tag,
        )


class MimoLocalEngine:
    """Adapt the upstream local ``MimoAudio`` object."""

    def __init__(self, mimo: Any) -> None:
        self.mimo = mimo

    def transcribe(self, audio_path: str, audio_tag: str) -> ASRResult:
        return ASRResult(
            text=self.mimo.asr_sft(audio_path, audio_tag=audio_tag),
            language=audio_tag,
        )
