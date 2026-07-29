#!/usr/bin/env python3
"""Unit tests for the MiMo HTTP API client. No real network is used."""

import base64
import json
import sys
import traceback
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from mimo_api import (  # noqa: E402
    MimoApiError,
    MimoApiRecognizer,
    MimoApiRetryableError,
    audio_tag_to_language,
    wav_data_url,
)


@pytest.mark.parametrize(
    ("audio_tag", "language"),
    [
        ("<chinese>", "zh"),
        ("<english>", "en"),
        ("<auto>", "auto"),
    ],
)
def test_audio_tag_language_mapping(audio_tag, language):
    assert audio_tag_to_language(audio_tag) == language


def test_wav_data_url(tmp_path):
    wav = tmp_path / "segment.wav"
    wav.write_bytes(b"RIFF-test-wav")
    assert wav_data_url(str(wav)) == (
        "data:audio/wav;base64,"
        + base64.b64encode(b"RIFF-test-wav").decode("ascii")
    )


def _recognizer(handler, *, api_key="top-secret", timeout=3):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return MimoApiRecognizer(
        api_key=api_key,
        base_url="https://api.example.test/v1/",
        model="mimo-test",
        timeout=timeout,
        client=client,
    )


def test_chat_completions_request_shape_and_api_key_header(tmp_path):
    wav = tmp_path / "segment.wav"
    wav.write_bytes(b"wav-data")
    captured = {}

    def handler(request):
        captured["request"] = request
        captured["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "  测试文本  "}}]},
        )

    text = _recognizer(handler).transcribe(str(wav), "<chinese>")
    request = captured["request"]
    payload = captured["json"]
    assert text == "测试文本"
    assert request.url.path == "/v1/chat/completions"
    assert request.headers["api-key"] == "top-secret"
    assert request.headers["content-type"] == "application/json"
    assert payload["model"] == "mimo-test"
    assert payload["asr_options"] == {"language": "zh"}
    assert payload["max_completion_tokens"] == 4096
    audio = payload["messages"][0]["content"][0]
    assert audio["type"] == "input_audio"
    assert audio["input_audio"]["data"].startswith("data:audio/wav;base64,")


def test_empty_content_falls_back_to_reasoning_content(tmp_path):
    wav = tmp_path / "segment.wav"
    wav.write_bytes(b"wav")

    def handler(_request):
        return httpx.Response(
            200,
            json={
                "choices": [{
                    "message": {
                        "content": "  ",
                        "reasoning_content": "  fallback text  ",
                    }
                }]
            },
        )

    assert _recognizer(handler).transcribe(str(wav), "<auto>") == "fallback text"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"choices": []},
        {"choices": [{"message": {"content": "", "reasoning_content": ""}}]},
    ],
)
def test_empty_or_malformed_response_raises(tmp_path, payload):
    wav = tmp_path / "segment.wav"
    wav.write_bytes(b"wav")

    def handler(_request):
        return httpx.Response(200, json=payload)

    with pytest.raises(MimoApiError, match=r"choices|transcription text"):
        _recognizer(handler).transcribe(str(wav), "<english>")


def test_invalid_json_raises_permanent_error(tmp_path):
    wav = tmp_path / "segment.wav"
    wav.write_bytes(b"wav")
    key = "json-secret"

    def handler(_request):
        return httpx.Response(200, content=f"not json {key}".encode())

    with pytest.raises(MimoApiError, match="valid JSON") as exc:
        _recognizer(handler, api_key=key).transcribe(str(wav), "<english>")
    assert key not in "".join(traceback.format_exception(exc.value))


@pytest.mark.parametrize("status", [401, 403])
def test_auth_errors_are_not_retryable_and_do_not_leak_key(tmp_path, status):
    wav = tmp_path / "segment.wav"
    wav.write_bytes(b"wav")
    key = "never-print-this-key"

    def handler(_request):
        return httpx.Response(status, text=key)

    with pytest.raises(MimoApiError) as exc_info:
        _recognizer(handler, api_key=key).transcribe(str(wav), "<chinese>")
    assert not isinstance(exc_info.value, MimoApiRetryableError)
    assert key not in str(exc_info.value)


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_transient_http_errors_are_retryable(tmp_path, status):
    wav = tmp_path / "segment.wav"
    wav.write_bytes(b"wav")

    def handler(_request):
        return httpx.Response(status)

    with pytest.raises(MimoApiRetryableError, match=str(status)):
        _recognizer(handler).transcribe(str(wav), "<chinese>")


def test_timeout_is_retryable_and_sanitized(tmp_path):
    wav = tmp_path / "segment.wav"
    wav.write_bytes(b"wav")
    key = "timeout-secret"

    def handler(request):
        raise httpx.ReadTimeout(key, request=request)

    with pytest.raises(MimoApiRetryableError, match="timed out") as exc_info:
        _recognizer(handler, api_key=key, timeout=7).transcribe(
            str(wav), "<chinese>"
        )
    assert key not in str(exc_info.value)
    assert key not in "".join(traceback.format_exception(exc_info.value))


def test_connection_error_is_retryable_and_sanitized(tmp_path):
    wav = tmp_path / "segment.wav"
    wav.write_bytes(b"wav")
    key = "connection-secret"

    def handler(request):
        raise httpx.ConnectError(key, request=request)

    with pytest.raises(MimoApiRetryableError, match="network request failed") as exc:
        _recognizer(handler, api_key=key).transcribe(str(wav), "<chinese>")
    assert key not in str(exc.value)
    assert key not in "".join(traceback.format_exception(exc.value))


def test_missing_audio_fails_without_request(tmp_path):
    calls = 0

    def handler(_request):
        nonlocal calls
        calls += 1
        return httpx.Response(200)

    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        _recognizer(handler).transcribe(
            str(tmp_path / "missing.wav"), "<chinese>"
        )
    assert calls == 0
