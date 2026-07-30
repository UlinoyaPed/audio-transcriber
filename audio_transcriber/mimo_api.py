#!/usr/bin/env python3
"""HTTP client for Xiaomi MiMo-V2.5-ASR.

The client performs one request per ``transcribe`` call. Retry policy belongs
to the orchestration layer in ``mimo_asr.py`` so this module remains easy to
unit test and cannot accidentally retry forever.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Optional

import httpx


_AUDIO_TAG_LANGUAGES = {
    "<chinese>": "zh",
    "<english>": "en",
    "<auto>": "auto",
}
_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
DEFAULT_MAX_AUDIO_BYTES = 20 * 1024 * 1024


class MimoApiError(RuntimeError):
    """A permanent MiMo API error that should not be retried."""


class MimoApiRetryableError(MimoApiError):
    """A transient MiMo API or network error that may be retried."""


def audio_tag_to_language(audio_tag: str) -> str:
    """Map the local MiMo audio tag syntax to the HTTP API language value."""
    try:
        return _AUDIO_TAG_LANGUAGES[audio_tag]
    except KeyError as exc:
        supported = ", ".join(_AUDIO_TAG_LANGUAGES)
        raise ValueError(
            f"Unsupported MiMo audio tag {audio_tag!r}; expected one of: {supported}"
        ) from exc


def wav_data_url(
    audio_path: str,
    *,
    max_bytes: int = DEFAULT_MAX_AUDIO_BYTES,
    read_chunk_size: int = 1024 * 1024,
) -> str:
    """Return a WAV file as a Base64 data URL with bounded raw-file reads.

    The MiMo chat-completions contract requires a JSON data URL, so the final
    encoded string must still exist in memory. Incremental encoding avoids a
    second, full-size raw audio copy and the size limit prevents accidentally
    creating an unbounded request.
    """
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if max_bytes <= 0:
        raise ValueError("MiMo API maximum audio size must be greater than zero")
    size = path.stat().st_size
    if size > max_bytes:
        raise MimoApiError(
            f"Audio file is too large for MiMo API JSON upload: "
            f"{size} bytes exceeds the configured {max_bytes}-byte limit"
        )
    if read_chunk_size <= 0:
        raise ValueError("read_chunk_size must be greater than zero")

    output = io.StringIO()
    output.write("data:audio/wav;base64,")
    remainder = b""
    with path.open("rb") as audio:
        while chunk := audio.read(read_chunk_size):
            data = remainder + chunk
            complete = len(data) - (len(data) % 3)
            if complete:
                output.write(base64.b64encode(data[:complete]).decode("ascii"))
            remainder = data[complete:]
    if remainder:
        output.write(base64.b64encode(remainder).decode("ascii"))
    return output.getvalue()


class MimoApiRecognizer:
    """Recognize one WAV segment through Xiaomi's chat-completions endpoint."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float,
        *,
        allow_reasoning_content: bool = False,
        max_audio_bytes: int = DEFAULT_MAX_AUDIO_BYTES,
        client: Optional[httpx.Client] = None,
    ) -> None:
        if not api_key:
            raise ValueError("MiMo API key is empty")
        if timeout <= 0:
            raise ValueError("MiMo API timeout must be greater than zero")
        if max_audio_bytes <= 0:
            raise ValueError("MiMo API maximum audio size must be greater than zero")
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = float(timeout)
        self.allow_reasoning_content = allow_reasoning_content
        self.max_audio_bytes = max_audio_bytes
        self._client = client

    def transcribe(self, audio_path: str, audio_tag: str) -> str:
        """Submit one WAV segment and return stripped transcript text."""
        data_url = wav_data_url(audio_path, max_bytes=self.max_audio_bytes)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": data_url},
                        }
                    ],
                }
            ],
            "asr_options": {"language": audio_tag_to_language(audio_tag)},
            "max_completion_tokens": 4096,
        }
        headers = {
            "Content-Type": "application/json",
            "api-key": self._api_key,
        }
        url = f"{self.base_url}/chat/completions"

        try:
            if self._client is not None:
                response = self._client.post(
                    url, headers=headers, json=payload, timeout=self.timeout
                )
            else:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(url, headers=headers, json=payload)
        except httpx.TimeoutException:
            raise MimoApiRetryableError(
                f"MiMo API request timed out after {self.timeout:g}s"
            ) from None
        except httpx.RequestError as exc:
            raise MimoApiRetryableError(
                f"MiMo API network request failed ({type(exc).__name__})"
            ) from None

        if response.status_code < 200 or response.status_code >= 300:
            error_cls = (
                MimoApiRetryableError
                if response.status_code in _RETRYABLE_STATUS_CODES
                else MimoApiError
            )
            raise error_cls(
                f"MiMo API returned HTTP {response.status_code}"
            )

        try:
            data: Any = response.json()
        except (ValueError, TypeError):
            raise MimoApiError("MiMo API response is not valid JSON") from None

        if not isinstance(data, dict):
            raise MimoApiError("MiMo API response must be a JSON object")
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise MimoApiError("MiMo API response does not contain choices[0]")
        first = choices[0]
        if not isinstance(first, dict) or not isinstance(first.get("message"), dict):
            raise MimoApiError(
                "MiMo API response does not contain choices[0].message"
            )

        message = first["message"]
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if self.allow_reasoning_content:
            reasoning = message.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning.strip():
                return reasoning.strip()
        suffix = (
            " or message.reasoning_content"
            if self.allow_reasoning_content
            else ""
        )
        raise MimoApiError(
            "MiMo API response contains no transcription text in "
            f"message.content{suffix}"
        )
