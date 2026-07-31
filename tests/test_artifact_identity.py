"""Offline artifact path and source-identity tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from audio_transcriber import transcribe


def test_default_outputs_are_isolated_beside_same_named_sources(tmp_path):
    first = tmp_path / "podcast-a" / "episode.flac"
    second = tmp_path / "podcast-b" / "episode.flac"
    assert transcribe.resolve_output_paths(first) == (
        first.parent / "episode_raw_transcript.json",
        first.parent / "episode-transcript.md",
    )
    assert transcribe.resolve_output_paths(second) == (
        second.parent / "episode_raw_transcript.json",
        second.parent / "episode-transcript.md",
    )


def test_raw_transcript_rejects_changed_source_or_parameters(tmp_path):
    audio = tmp_path / "episode.flac"
    audio.write_bytes(b"first recording")
    path = tmp_path / "episode_raw_transcript.json"
    processing = {"lang": "zh", "asr_model": "model-a"}
    document = transcribe.build_raw_transcript_document(
        audio,
        [{"speaker": 0, "start_ms": 0, "end_ms": 1, "text": "hello"}],
        processing,
    )
    path.write_text(json.dumps(document), encoding="utf-8")

    segments, loaded = transcribe.load_raw_transcript_document(
        path,
        audio_path=audio,
        expected_processing=processing,
        require_identity=True,
    )
    assert segments[0]["text"] == "hello"
    assert loaded["source_audio"]["sha256"]

    audio.write_bytes(b"different recording")
    with pytest.raises(RuntimeError, match="different or modified"):
        transcribe.load_raw_transcript_document(
            path,
            audio_path=audio,
            expected_processing=processing,
            require_identity=True,
        )
    with pytest.raises(RuntimeError, match="processing parameters"):
        transcribe.load_raw_transcript_document(
            path,
            expected_processing={"lang": "en"},
            require_identity=True,
        )


def test_legacy_raw_json_is_not_silently_reused(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(RuntimeError, match="legacy identity-free"):
        transcribe.load_raw_transcript_document(path, require_identity=True)


def test_existing_outputs_require_explicit_overwrite(tmp_path):
    audio = tmp_path / "episode.wav"
    audio.write_bytes(b"audio")
    (tmp_path / "episode_raw_transcript.json").write_text(
        "do not replace", encoding="utf-8"
    )
    with patch.object(
        sys,
        "argv",
        ["audio-transcriber", str(audio), "--skip-llm", "--device", "cpu"],
    ):
        with pytest.raises(SystemExit) as exc:
            transcribe.main()
    assert exc.value.code == 2
    assert (
        tmp_path / "episode_raw_transcript.json"
    ).read_text(encoding="utf-8") == "do not replace"
