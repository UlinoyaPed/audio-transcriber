"""LLM cleanup cache identity and corruption tests."""

from __future__ import annotations

import json

import pytest

from audio_transcriber import transcribe


def _segment(text="原始文本"):
    return [{"speaker": 0, "start_ms": 0, "end_ms": 1000, "text": text}]


def _run(cache_dir, *, merged=None, speaker_map=None, **overrides):
    options = {
        "merged": merged or _segment(),
        "speaker_map": speaker_map or {0: "Alice"},
        "model_id": "model-a",
        "region": "region-a",
        "speaker_context": {"Alice": "host"},
        "cache_dir": cache_dir,
        "reference_text": "reference-a",
        "speaker_names": ["Alice"],
        "speaker_genders": {"Alice": "female"},
        "provider": "openai",
    }
    options.update(overrides)
    return transcribe.run_llm_cleanup(**options)


def test_llm_cache_reuses_only_exact_fingerprint(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(transcribe.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        transcribe,
        "call_llm",
        lambda *_args, **_kwargs: calls.append("call") or "清理结果",
    )

    assert _run(tmp_path) == ["清理结果"]
    assert _run(tmp_path) == ["清理结果"]
    assert calls == ["call"]
    cache = json.loads((tmp_path / "chunk_000.json").read_text())
    assert cache["schema_version"] == 1
    assert len(cache["fingerprint"]) == 64
    assert cache["result"] == "清理结果"


@pytest.mark.parametrize(
    "change",
    [
        {"model_id": "model-b"},
        {"provider": "anthropic"},
        {"reference_text": "reference-b"},
        {"speaker_context": {"Alice": "guest"}},
        {"speaker_names": ["Alicia"]},
        {"speaker_genders": {"Alice": "male"}},
        {"speaker_map": {0: "Alicia"}},
        {"merged": _segment("修正后的原始文本")},
    ],
)
def test_llm_cache_invalidates_on_semantic_input_change(
    tmp_path, monkeypatch, change
):
    calls = []
    monkeypatch.setattr(transcribe.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        transcribe,
        "call_llm",
        lambda *_args, **_kwargs: calls.append("call") or f"结果{len(calls)}",
    )

    _run(tmp_path)
    _run(tmp_path, **change)
    assert calls == ["call", "call"]


def test_truncated_llm_cache_is_rebuilt_atomically(tmp_path, monkeypatch):
    cache_file = tmp_path / "chunk_000.json"
    cache_file.write_text('{"schema_version":', encoding="utf-8")
    monkeypatch.setattr(transcribe.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        transcribe, "call_llm", lambda *_args, **_kwargs: "完整结果"
    )

    assert _run(tmp_path) == ["完整结果"]
    parsed = json.loads(cache_file.read_text(encoding="utf-8"))
    assert parsed["result"] == "完整结果"
    assert not list(tmp_path.glob("*.tmp"))
