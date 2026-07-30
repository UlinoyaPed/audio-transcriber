"""Tests for the normalized ASR backend interface."""

from audio_transcriber.asr_engine import ASRResult, MimoApiEngine, MimoLocalEngine


def test_mimo_api_engine_returns_normalized_result():
    class Recognizer:
        def transcribe(self, audio_path, audio_tag):
            assert audio_path == "segment.wav"
            assert audio_tag == "<auto>"
            return "测试文本"

    result = MimoApiEngine(Recognizer()).transcribe("segment.wav", "<auto>")
    assert result == ASRResult(text="测试文本", language="<auto>")


def test_mimo_local_engine_preserves_upstream_call():
    class LocalMimo:
        def asr_sft(self, audio_path, *, audio_tag):
            assert audio_path == "segment.wav"
            assert audio_tag == "<chinese>"
            return "本地文本"

    result = MimoLocalEngine(LocalMimo()).transcribe(
        "segment.wav", "<chinese>"
    )
    assert result.text == "本地文本"
