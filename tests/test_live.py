"""Tests for live recording/transcription without a microphone or network."""

from __future__ import annotations

import json
import threading
import time
import wave
from pathlib import Path

import pytest

from audio_transcriber.live import (
    AppendOnlyJournal,
    PcmChunkWriter,
    RecordedChunk,
    SerialTranscriptionWorker,
    StreamingMimoApiProcessor,
    finalize_live_output,
    list_input_devices,
    recover_live_checkpoint,
    save_live_checkpoint,
)
from audio_transcriber.mimo_api import MimoApiError, MimoApiRetryableError
from audio_transcriber.transcribe import load_raw_transcript_document


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


def test_append_only_journal_is_valid_jsonl_and_redacts_key(tmp_path):
    key = "journal-secret"
    journal_path = tmp_path / "session.jsonl"
    journal = AppendOnlyJournal(journal_path, secrets=(key,))
    journal.append("session_started", model="mimo", detail=f"contains {key}")
    journal.append("chunk_committed", index=0)

    raw = journal_path.read_text(encoding="utf-8")
    records = [json.loads(line) for line in raw.splitlines()]
    assert key not in raw
    assert [record["event"] for record in records] == [
        "session_started",
        "chunk_committed",
    ]
    assert records[0]["detail"] == "contains [REDACTED]"


def test_append_only_journal_serializes_concurrent_writers(tmp_path):
    journal_path = tmp_path / "concurrent.jsonl"
    journal = AppendOnlyJournal(journal_path)

    def writer(worker):
        for index in range(50):
            journal.append("progress", worker=worker, index=index)

    threads = [threading.Thread(target=writer, args=(worker,)) for worker in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    records = [
        json.loads(line)
        for line in journal_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 200
    assert {
        (record["worker"], record["index"]) for record in records
    } == {(worker, index) for worker in range(4) for index in range(50)}


def test_live_checkpoint_paths_are_relative_and_fsyncd(tmp_path, monkeypatch):
    checkpoint = tmp_path / "session" / "meeting_live_partial.json"
    checkpoint.parent.mkdir()
    calls = []
    monkeypatch.setattr("audio_transcriber.live.os.fsync", lambda fd: calls.append(fd))
    save_live_checkpoint(
        checkpoint,
        status="recording",
        recording_path=checkpoint.parent / "meeting.wav",
        model="mimo-v2.5-asr",
        base_url="https://api.example/v1",
        audio_tag="<auto>",
        chunk_seconds=15,
        completed_chunks=[],
        segments=[],
        journal_path=checkpoint.parent / "meeting_live_journal.jsonl",
        output_paths={
            "raw_json": str(checkpoint.parent / "meeting_raw_transcript.json"),
            "markdown": str(checkpoint.parent / "meeting-transcript.md"),
        },
    )
    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert state["recording_path"] == "meeting.wav"
    assert state["journal_path"] == "meeting_live_journal.jsonl"
    assert state["output_paths"]["raw_json"] == "meeting_raw_transcript.json"
    assert len(calls) >= 2


def test_recovery_resolves_relative_paths_from_checkpoint_directory(
    tmp_path, monkeypatch
):
    session = tmp_path / "session"
    elsewhere = tmp_path / "elsewhere"
    session.mkdir()
    elsewhere.mkdir()
    recording = session / "meeting.wav"
    _write_chunk(recording, _pcm(2))
    checkpoint = session / "meeting_live_partial.json"
    checkpoint.write_text(
        json.dumps(
            {
                "version": 1,
                "status": "failed",
                "backend": "api",
                "model": "mimo-v2.5-asr",
                "base_url": "https://api.example/v1",
                "audio_tag": "<auto>",
                "recording_path": "meeting.wav",
                "journal_path": "meeting_live_journal.jsonl",
                "output_paths": {
                    "raw_json": "meeting_raw_transcript.json",
                    "markdown": "meeting-transcript.md",
                },
                "segments": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(elsewhere)

    def transcribe(audio_path, **_kwargs):
        assert audio_path == str(recording)
        return [
            {
                "speaker": 0,
                "start_ms": 0,
                "end_ms": 1000,
                "text": "跨目录恢复",
            }
        ]

    outputs = recover_live_checkpoint(
        checkpoint,
        api_key="secret",
        device="cpu",
        title="恢复",
        api_timeout=30,
        allow_reasoning_content=False,
        max_audio_bytes=1024,
        transcribe_fn=transcribe,
    )
    assert outputs == (
        session / "meeting_raw_transcript.json",
        session / "meeting-transcript.md",
    )


def test_recover_checkpoint_reprocesses_master_wav_without_microphone(tmp_path):
    recording = tmp_path / "meeting.wav"
    _write_chunk(recording, _pcm(5))
    checkpoint = tmp_path / "meeting_live_partial.json"
    journal = tmp_path / "meeting_live_journal.jsonl"
    raw_json = tmp_path / "result.json"
    markdown = tmp_path / "result.md"
    checkpoint.write_text(
        json.dumps(
            {
                "version": 1,
                "status": "failed",
                "backend": "api",
                "model": "mimo-v2.5-asr",
                "base_url": "https://api.example/v1",
                "audio_tag": "<auto>",
                "recording_path": str(recording),
                "num_speakers": 1,
                "speaker_names": ["张三"],
                "journal_path": str(journal),
                "output_paths": {
                    "raw_json": str(raw_json),
                    "markdown": str(markdown),
                },
                "segments": [],
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    def transcribe(audio_path, **kwargs):
        captured["audio_path"] = audio_path
        captured.update(kwargs)
        return [
            {
                "speaker": 0,
                "start_ms": 1000,
                "end_ms": 5000,
                "text": "恢复文本",
            }
        ]

    outputs = recover_live_checkpoint(
        checkpoint,
        api_key="recovery-secret",
        device="cpu",
        title="恢复会议",
        api_timeout=30,
        allow_reasoning_content=False,
        max_audio_bytes=1024,
        transcribe_fn=transcribe,
    )

    assert outputs == (raw_json, markdown)
    assert captured["audio_path"] == str(recording)
    assert captured["backend"] == "api"
    assert captured["device"] == "cpu"
    raw_document = json.loads(raw_json.read_text(encoding="utf-8"))
    assert raw_document["segments"][0] == {
        "speaker": 0,
        "start_ms": 1000,
        "end_ms": 5000,
        "text": "恢复文本",
    }
    assert raw_document["source_audio"]["path"] == str(recording.resolve())
    assert raw_document["processing"]["mimo_backend"] == "api"
    assert "MiMo API (mimo-v2.5-asr)" in markdown.read_text(encoding="utf-8")
    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert state["status"] == "complete"
    assert "recovery-secret" not in checkpoint.read_text(encoding="utf-8")
    assert json.loads(journal.read_text(encoding="utf-8"))["event"] == (
        "session_recovered"
    )


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
        base_url="https://api.xiaomimimo.com/v1",
        audio_tag="<auto>",
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
    raw_document = json.loads(raw_json.read_text(encoding="utf-8"))
    assert raw_document["segments"] == result
    assert raw_document["source_audio"]["sha256"]
    assert raw_document["processing"]["mimo_api_model"] == "mimo-v2.5-asr"
    assert raw_document["processing"]["num_speakers"] == 1
    loaded, _ = load_raw_transcript_document(
        raw_json,
        audio_path=recording,
        expected_processing=raw_document["processing"],
        require_identity=True,
    )
    assert loaded == result
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
