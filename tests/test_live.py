"""Tests for live recording/transcription without a microphone or network."""

from __future__ import annotations

import json
import threading
import time
import wave
from pathlib import Path

import pytest

from audio_transcriber.live import (
    PcmChunkWriter,
    RecordedChunk,
    SerialTranscriptionWorker,
    StreamingMimoApiProcessor,
    finalize_live_output,
    list_input_devices,
    save_live_checkpoint,
)
from audio_transcriber.mimo_api import MimoApiError, MimoApiRetryableError


def _pcm(seconds: float, sample_rate: int = 16_000) -> bytes:
    return b"\x00\x00" * round(seconds * sample_rate)


def _write_chunk(path: Path, pcm: bytes, sample_rate: int = 16_000) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)


class SequenceVad:
    def __init__(self, values):
        self.values = list(values)
        self.calls = 0

    def generate(self, *, input):
        with wave.open(input, "rb") as wav:
            assert wav.getframerate() == 16_000
        value = self.values[self.calls]
        self.calls += 1
        return [{"value": value}]


class FakeRecognizer:
    def __init__(self, texts=None, error=None):
        self.texts = list(texts or [])
        self.error = error
        self.calls = []

    def transcribe(self, audio_path, audio_tag):
        self.calls.append((audio_path, audio_tag, threading.get_ident()))
        if self.error is not None:
            raise self.error
        return self.texts.pop(0)


def test_pcm_chunk_writer_preserves_master_and_boundaries(tmp_path):
    recording = tmp_path / "recording.wav"
    chunk_dir = tmp_path / "chunks"
    writer = PcmChunkWriter(recording, chunk_dir, chunk_seconds=1)

    assert writer.write(_pcm(0.6)) == []
    # The master header is patched continuously, not only on graceful close.
    with wave.open(str(recording), "rb") as in_progress:
        assert in_progress.getnframes() == round(0.6 * 16_000)
    emitted = writer.write(_pcm(1.7))
    tail = writer.close()

    assert [(chunk.start_ms, chunk.end_ms) for chunk in emitted + tail] == [
        (0, 1000),
        (1000, 2000),
        (2000, 2300),
    ]
    with wave.open(str(recording), "rb") as wav:
        assert wav.getnframes() == round(2.3 * 16_000)
    assert sum(
        wave.open(chunk.path, "rb").getnframes() for chunk in emitted + tail
    ) == round(2.3 * 16_000)


def test_streaming_processor_carries_boundary_speech(tmp_path):
    first_path = tmp_path / "first.wav"
    second_path = tmp_path / "second.wav"
    _write_chunk(first_path, _pcm(2))
    _write_chunk(second_path, _pcm(2))
    vad = SequenceVad(
        [
            [[1000, 2000]],  # touches edge: carry, do not recognize yet
            [[0, 2200]],  # combined buffer starts at absolute 1000ms
        ]
    )
    recognizer = FakeRecognizer(["完整句子"])
    processor = StreamingMimoApiProcessor(
        recognizer,
        vad,
        audio_tag="<chinese>",
        api_key="secret",
        backoffs=(),
    )
    try:
        assert processor.feed_chunk(
            RecordedChunk(0, str(first_path), 0, 2000)
        ) == []
        result = processor.feed_chunk(
            RecordedChunk(1, str(second_path), 2000, 4000)
        )
    finally:
        processor.close()

    assert result == [
        {
            "idx": 0,
            "start_ms": 1000,
            "end_ms": 3200,
            "text": "完整句子",
        }
    ]
    assert len(recognizer.calls) == 1
    assert recognizer.calls[0][1] == "<chinese>"


def test_streaming_processor_flushes_final_boundary_segment(tmp_path):
    path = tmp_path / "only.wav"
    _write_chunk(path, _pcm(2))
    vad = SequenceVad([[[500, 2000]], [[0, 1500]]])
    recognizer = FakeRecognizer(["结尾"])
    processor = StreamingMimoApiProcessor(
        recognizer,
        vad,
        audio_tag="<auto>",
        api_key="secret",
        backoffs=(),
    )
    try:
        assert processor.feed_chunk(RecordedChunk(0, str(path), 0, 2000)) == []
        result = processor.flush()
    finally:
        processor.close()
    assert result[0]["start_ms"] == 500
    assert result[0]["end_ms"] == 2000
    assert result[0]["text"] == "结尾"


def test_streaming_processor_error_has_time_but_not_api_key(tmp_path):
    path = tmp_path / "chunk.wav"
    _write_chunk(path, _pcm(2))
    key = "do-not-print"
    vad = SequenceVad([[[200, 1200]]])
    recognizer = FakeRecognizer(error=MimoApiError(f"bad request {key}"))
    processor = StreamingMimoApiProcessor(
        recognizer,
        vad,
        audio_tag="<english>",
        api_key=key,
        backoffs=(),
    )
    try:
        with pytest.raises(RuntimeError) as exc:
            processor.feed_chunk(RecordedChunk(0, str(path), 0, 2000))
    finally:
        processor.close()
    message = str(exc.value)
    assert "200ms-1200ms" in message
    assert key not in message
    assert len(recognizer.calls) == 1


def test_streaming_processor_retries_transient_api_error(tmp_path):
    path = tmp_path / "chunk.wav"
    _write_chunk(path, _pcm(2))
    vad = SequenceVad([[[200, 1200]]])

    class FlakyRecognizer(FakeRecognizer):
        def transcribe(self, audio_path, audio_tag):
            self.calls.append((audio_path, audio_tag, threading.get_ident()))
            if len(self.calls) == 1:
                raise MimoApiRetryableError("HTTP 429")
            return "重试成功"

    recognizer = FlakyRecognizer()
    processor = StreamingMimoApiProcessor(
        recognizer,
        vad,
        audio_tag="<auto>",
        api_key="secret",
        backoffs=(0,),
    )
    try:
        result = processor.feed_chunk(RecordedChunk(0, str(path), 0, 2000))
    finally:
        processor.close()
    assert result[0]["text"] == "重试成功"
    assert len(recognizer.calls) == 2


def test_serial_worker_preserves_order_and_uses_one_thread():
    class Processor:
        def __init__(self):
            self.active = 0
            self.max_active = 0
            self.thread_ids = set()

        def feed_chunk(self, chunk):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.thread_ids.add(threading.get_ident())
            time.sleep(0.01)
            self.active -= 1
            return [
                {
                    "idx": chunk.index,
                    "start_ms": chunk.start_ms,
                    "end_ms": chunk.end_ms,
                    "text": str(chunk.index),
                }
            ]

        def flush(self):
            return []

    processor = Processor()
    worker = SerialTranscriptionWorker(processor)
    worker.start()
    for index in range(3):
        worker.submit(RecordedChunk(index, f"{index}.wav", index, index + 1))
    result = worker.finish()

    assert [segment["idx"] for segment in result] == [0, 1, 2]
    assert worker.completed_chunks == [0, 1, 2]
    assert processor.max_active == 1
    assert len(processor.thread_ids) == 1


def test_checkpoint_schema_redacts_key(tmp_path):
    checkpoint = tmp_path / "live_partial.json"
    key = "checkpoint-secret"
    save_live_checkpoint(
        checkpoint,
        status="failed",
        recording_path=tmp_path / "live.wav",
        model="mimo-v2.5-asr",
        base_url="https://api.example/v1",
        audio_tag="<auto>",
        chunk_seconds=15,
        completed_chunks=[0],
        segments=[
            {
                "idx": 0,
                "start_ms": 0,
                "end_ms": 1000,
                "text": f"text {key}",
            }
        ],
        failed_at={"error": f"request failed {key}"},
        secrets=(key,),
    )
    raw = checkpoint.read_text(encoding="utf-8")
    state = json.loads(raw)
    assert key not in raw
    assert state["backend"] == "api"
    assert state["model"] == "mimo-v2.5-asr"
    assert state["base_url"] == "https://api.example/v1"
    assert state["segments"][0]["text"] == "text [REDACTED]"


def test_finalize_live_output_runs_cam_and_writes_unified_outputs(tmp_path):
    recording = tmp_path / "meeting.wav"
    _write_chunk(recording, _pcm(5))
    raw_json = tmp_path / "meeting_raw_transcript.json"
    markdown = tmp_path / "meeting-transcript.md"
    source = [
        {"idx": 0, "start_ms": 1000, "end_ms": 5000, "text": "测试文本"}
    ]
    captured = {}

    def assigner(segments, audio_path, num_speakers, spk_model_id, device):
        captured.update(
            {
                "audio_path": audio_path,
                "num_speakers": num_speakers,
                "device": device,
                "spk_model_id": spk_model_id,
            }
        )
        return [
            {
                "speaker": 0,
                "start_ms": item["start_ms"],
                "end_ms": item["end_ms"],
                "text": item["text"],
            }
            for item in segments
        ]

    result = finalize_live_output(
        recording,
        source,
        raw_json_path=raw_json,
        markdown_path=markdown,
        num_speakers=1,
        speaker_names=["张三"],
        device="cpu",
        model="mimo-v2.5-asr",
        title="实时会议",
        assign_speakers_fn=assigner,
    )

    assert result == [
        {
            "speaker": 0,
            "start_ms": 1000,
            "end_ms": 5000,
            "text": "测试文本",
        }
    ]
    assert json.loads(raw_json.read_text(encoding="utf-8")) == result
    rendered = markdown.read_text(encoding="utf-8")
    assert "MiMo API (mimo-v2.5-asr)" in rendered
    assert "[00:00:01] 张三: 测试文本" in rendered
    assert captured["device"] == "cpu"
    assert captured["num_speakers"] == 1


def test_list_input_devices_uses_injected_module(capsys):
    class FakeSoundDevice:
        @staticmethod
        def query_devices():
            return "0 Built-in microphone"

    list_input_devices(FakeSoundDevice)
    assert "Built-in microphone" in capsys.readouterr().out
