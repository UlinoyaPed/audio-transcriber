"""Audio preprocessing identity and atomic-output tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

from audio_transcriber import transcribe


def _install_fake_ffmpeg(monkeypatch, calls):
    monkeypatch.setattr(transcribe.shutil, "which", lambda _tool: "/bin/tool")
    monkeypatch.setattr(transcribe, "get_audio_duration", lambda _path: 30.0)
    monkeypatch.setattr(
        transcribe,
        "_is_16k_mono",
        lambda path: ".tmp-" in path or ".transcoded." in path
        or path.endswith("episode.flac"),
    )

    def fake_run(cmd, **_kwargs):
        calls.append(list(cmd))
        if cmd[0] == "ffmpeg":
            with open(cmd[-2], "wb") as output:
                output.write(b"converted-16k-mono")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(transcribe.subprocess, "run", fake_run)


def test_same_suffix_flac_uses_distinct_atomic_target(tmp_path, monkeypatch):
    source = tmp_path / "episode.flac"
    source.write_bytes(b"original-48k-stereo")
    calls = []
    _install_fake_ffmpeg(monkeypatch, calls)
    monkeypatch.setattr(
        transcribe,
        "_is_16k_mono",
        lambda path: path != str(source),
    )

    result = transcribe.preprocess_audio(str(source), "flac")

    output = tmp_path / "episode.transcoded.flac"
    assert result == str(output)
    assert source.read_bytes() == b"original-48k-stereo"
    assert output.read_bytes() == b"converted-16k-mono"
    assert len([cmd for cmd in calls if cmd[0] == "ffmpeg"]) == 1
    manifest = json.loads(
        (tmp_path / "episode.transcoded.flac.manifest.json").read_text()
    )
    assert manifest["source"]["path"] == str(source.resolve())
    assert manifest["parameters"]["sample_rate"] == 16_000
    assert manifest["parameters"]["channels"] == 1


def test_conversion_cache_reuses_only_matching_source_identity(
    tmp_path, monkeypatch
):
    source = tmp_path / "episode.mp3"
    source.write_bytes(b"first-program")
    calls = []
    _install_fake_ffmpeg(monkeypatch, calls)

    first = transcribe.preprocess_audio(str(source), "flac")
    second = transcribe.preprocess_audio(str(source), "flac")
    assert first == second
    assert len([cmd for cmd in calls if cmd[0] == "ffmpeg"]) == 1

    source.write_bytes(b"a-different-program")
    third = transcribe.preprocess_audio(str(source), "flac")
    assert third == first
    assert len([cmd for cmd in calls if cmd[0] == "ffmpeg"]) == 2
    manifest = json.loads(
        (tmp_path / "episode.flac.manifest.json").read_text()
    )
    assert manifest["source"]["sha256"] == transcribe._sha256_file(source)
