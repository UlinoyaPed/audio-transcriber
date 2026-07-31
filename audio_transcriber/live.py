#!/usr/bin/env python3
"""Microphone recording with serial, incremental MiMo API transcription.

The capture callback never performs network or model work. It copies PCM into
a bounded queue; the foreground recorder drains that queue to a durable master
WAV and fixed-size spool files. A single worker recognizes those files in
order. Speaker clustering runs once against the complete recording so speaker
IDs stay stable across chunk boundaries.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from .asr_engine import ASREngine, MimoApiEngine
from .mimo_api import MimoApiRecognizer, MimoApiRetryableError
from .mimo_asr import (
    _safe_error_text,
    assign_speakers_via_cam,
    recognize_with_retry,
)
from .model_revisions import modelscope_revision
from .transcribe import (
    MODEL_PRESETS,
    assemble_markdown,
    build_speaker_map,
    chunk_by_duration,
    format_chunk,
    merge_consecutive,
    resolve_mimo_api_config,
)


SAMPLE_RATE = 16_000
CHANNELS = 1
SAMPLE_WIDTH = 2


class LiveTranscriptionError(RuntimeError):
    """A recording or recognition failure with actionable context."""


@dataclass(frozen=True)
class RecordedChunk:
    """A durable, sequential PCM chunk from the master recording."""

    index: int
    path: str
    start_ms: int
    end_ms: int


def _write_pcm_wav(
    path: Path,
    pcm: bytes,
    *,
    sample_rate: int = SAMPLE_RATE,
) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)


def _read_pcm_wav(
    path: str,
    *,
    expected_sample_rate: int = SAMPLE_RATE,
) -> bytes:
    with wave.open(path, "rb") as wav:
        if wav.getnchannels() != CHANNELS:
            raise LiveTranscriptionError(
                f"Live chunk must be mono: {path} has {wav.getnchannels()} channels"
            )
        if wav.getsampwidth() != SAMPLE_WIDTH:
            raise LiveTranscriptionError(
                f"Live chunk must be 16-bit PCM: {path} has "
                f"{wav.getsampwidth() * 8}-bit samples"
            )
        if wav.getframerate() != expected_sample_rate:
            raise LiveTranscriptionError(
                f"Live chunk sample rate mismatch: expected "
                f"{expected_sample_rate}, got {wav.getframerate()}"
            )
        return wav.readframes(wav.getnframes())


class PcmChunkWriter:
    """Write one master WAV and emit disk-backed chunks without losing order."""

    def __init__(
        self,
        recording_path: Path,
        chunk_dir: Path,
        *,
        chunk_seconds: float,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        if chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be greater than zero")
        self.recording_path = Path(recording_path)
        self.chunk_dir = Path(chunk_dir)
        self.sample_rate = sample_rate
        self.bytes_per_frame = CHANNELS * SAMPLE_WIDTH
        self.chunk_frames = max(1, round(chunk_seconds * sample_rate))
        self.chunk_bytes = self.chunk_frames * self.bytes_per_frame
        self._buffer = bytearray()
        self._emitted_frames = 0
        self._next_index = 0
        self._closed = False

        self.recording_path.parent.mkdir(parents=True, exist_ok=True)
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        # Use an unbuffered handle and wave.writeframes() so the RIFF sizes are
        # patched after every microphone block. The master WAV remains readable
        # up to the latest drained block even if the process exits unexpectedly.
        self._master_file = self.recording_path.open("w+b", buffering=0)
        self._master = wave.open(self._master_file, "wb")
        self._master.setnchannels(CHANNELS)
        self._master.setsampwidth(SAMPLE_WIDTH)
        self._master.setframerate(sample_rate)

    def write(self, pcm: bytes) -> list[RecordedChunk]:
        if self._closed:
            raise RuntimeError("cannot write after PcmChunkWriter.close()")
        if len(pcm) % self.bytes_per_frame:
            raise LiveTranscriptionError(
                f"Microphone returned {len(pcm)} bytes, which is not aligned "
                f"to {self.bytes_per_frame}-byte audio frames"
            )
        self._master.writeframes(pcm)
        self._buffer.extend(pcm)
        emitted: list[RecordedChunk] = []
        while len(self._buffer) >= self.chunk_bytes:
            chunk_pcm = bytes(self._buffer[: self.chunk_bytes])
            del self._buffer[: self.chunk_bytes]
            emitted.append(self._emit(chunk_pcm))
        return emitted

    def close(self) -> list[RecordedChunk]:
        if self._closed:
            return []
        emitted = []
        if self._buffer:
            emitted.append(self._emit(bytes(self._buffer)))
            self._buffer.clear()
        self._master.close()
        self._master_file.close()
        self._closed = True
        return emitted

    def _emit(self, pcm: bytes) -> RecordedChunk:
        frames = len(pcm) // self.bytes_per_frame
        start_frame = self._emitted_frames
        end_frame = start_frame + frames
        path = self.chunk_dir / f"chunk_{self._next_index:06d}.wav"
        _write_pcm_wav(path, pcm, sample_rate=self.sample_rate)
        chunk = RecordedChunk(
            index=self._next_index,
            path=str(path),
            start_ms=round(start_frame * 1000 / self.sample_rate),
            end_ms=round(end_frame * 1000 / self.sample_rate),
        )
        self._next_index += 1
        self._emitted_frames = end_frame
        return chunk


def _vad_intervals(vad_model: Any, audio_path: str) -> list[tuple[int, int]]:
    result = vad_model.generate(input=audio_path)
    if not result or not isinstance(result, list):
        return []
    first = result[0]
    if not isinstance(first, dict) or not isinstance(first.get("value"), list):
        return []
    intervals = []
    for value in first["value"]:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            intervals.append((int(value[0]), int(value[1])))
    return intervals


class StreamingMimoApiProcessor:
    """Run cached FSMN VAD and serial MiMo API recognition over live chunks.

    A VAD segment touching the right edge is carried into the next chunk. This
    avoids cutting ordinary utterances at an arbitrary timer boundary. The
    carry buffer is bounded; very long uninterrupted speech is eventually
    forced through to keep latency and memory finite.
    """

    def __init__(
        self,
        recognizer: ASREngine,
        vad_model: Any,
        *,
        audio_tag: str,
        api_key: str,
        boundary_guard_ms: int = 350,
        max_pending_seconds: float = 60,
        backoffs: Sequence[float] = (1.0, 2.0, 5.0, 10.0),
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        if boundary_guard_ms < 0:
            raise ValueError("boundary_guard_ms must not be negative")
        if max_pending_seconds <= 0:
            raise ValueError("max_pending_seconds must be greater than zero")
        self.recognizer = recognizer
        self.vad_model = vad_model
        self.audio_tag = audio_tag
        self._api_key = api_key
        self.boundary_guard_ms = boundary_guard_ms
        self.max_pending_ms = round(max_pending_seconds * 1000)
        self.backoffs = tuple(backoffs)
        self.sample_rate = sample_rate
        self._pending_pcm = b""
        self._pending_start_ms: Optional[int] = None
        self._last_chunk_end_ms: Optional[int] = None
        self._next_segment_index = 0
        self._temp_dir = tempfile.TemporaryDirectory(prefix="audio-transcriber-live-")

    def close(self) -> None:
        self._temp_dir.cleanup()

    def feed_chunk(self, chunk: RecordedChunk) -> list[dict]:
        if (
            self._last_chunk_end_ms is not None
            and abs(chunk.start_ms - self._last_chunk_end_ms) > 1
        ):
            raise LiveTranscriptionError(
                f"Live chunk sequence has a gap or overlap: chunk {chunk.index} "
                f"starts at {chunk.start_ms}ms, expected "
                f"{self._last_chunk_end_ms}ms"
            )
        pcm = _read_pcm_wav(chunk.path, expected_sample_rate=self.sample_rate)
        self._last_chunk_end_ms = chunk.end_ms
        if self._pending_pcm:
            assert self._pending_start_ms is not None
            buffer_start_ms = self._pending_start_ms
            pcm = self._pending_pcm + pcm
        else:
            buffer_start_ms = chunk.start_ms
        return self._process_buffer(pcm, buffer_start_ms, final=False)

    def flush(self) -> list[dict]:
        if not self._pending_pcm:
            return []
        assert self._pending_start_ms is not None
        pcm = self._pending_pcm
        start_ms = self._pending_start_ms
        self._pending_pcm = b""
        self._pending_start_ms = None
        return self._process_buffer(pcm, start_ms, final=True)

    def _process_buffer(
        self,
        pcm: bytes,
        buffer_start_ms: int,
        *,
        final: bool,
    ) -> list[dict]:
        buffer_path = Path(self._temp_dir.name) / "vad_buffer.wav"
        _write_pcm_wav(buffer_path, pcm, sample_rate=self.sample_rate)
        duration_ms = round(
            len(pcm) * 1000 / (self.sample_rate * CHANNELS * SAMPLE_WIDTH)
        )
        intervals = [
            (max(0, start), min(duration_ms, end))
            for start, end in _vad_intervals(self.vad_model, str(buffer_path))
            if end > start and end > 0 and start < duration_ms
        ]

        self._pending_pcm = b""
        self._pending_start_ms = None
        if not intervals:
            return []

        stable = intervals
        touches_boundary = (
            intervals[-1][1] >= duration_ms - self.boundary_guard_ms
        )
        if not final and touches_boundary and duration_ms < self.max_pending_ms:
            pending_start = intervals[-1][0]
            pending_frame = round(pending_start * self.sample_rate / 1000)
            self._pending_pcm = pcm[pending_frame * SAMPLE_WIDTH :]
            self._pending_start_ms = buffer_start_ms + pending_start
            stable = intervals[:-1]

        recognized = []
        for local_start_ms, local_end_ms in stable:
            start_frame = round(local_start_ms * self.sample_rate / 1000)
            end_frame = round(local_end_ms * self.sample_rate / 1000)
            segment_pcm = pcm[
                start_frame * SAMPLE_WIDTH : end_frame * SAMPLE_WIDTH
            ]
            absolute_start = buffer_start_ms + local_start_ms
            absolute_end = buffer_start_ms + local_end_ms
            segment_path = (
                Path(self._temp_dir.name)
                / f"segment_{self._next_segment_index:06d}.wav"
            )
            _write_pcm_wav(segment_path, segment_pcm, sample_rate=self.sample_rate)
            try:
                def recognize(path: str) -> str:
                    result = self.recognizer.transcribe(path, self.audio_tag)
                    # Retain compatibility with third-party/fake recognizers
                    # that predate the normalized ASREngine interface.
                    return result.text if hasattr(result, "text") else str(result)

                text = recognize_with_retry(
                    recognize,
                    str(segment_path),
                    max_attempts=len(self.backoffs) + 1,
                    backoffs=self.backoffs,
                    is_retryable=lambda exc: isinstance(
                        exc, MimoApiRetryableError
                    ),
                    secrets=(self._api_key,),
                )
            except RuntimeError as exc:
                safe_error = _safe_error_text(exc, (self._api_key,))
                raise LiveTranscriptionError(
                    f"Live segment {self._next_segment_index} failed at "
                    f"{absolute_start}ms-{absolute_end}ms: {safe_error}"
                ) from exc
            if self._api_key and self._api_key in text:
                text = text.replace(self._api_key, "[REDACTED]")
            recognized.append(
                {
                    "idx": self._next_segment_index,
                    "start_ms": absolute_start,
                    "end_ms": absolute_end,
                    "text": text.strip(),
                }
            )
            self._next_segment_index += 1
        return recognized


ProgressCallback = Callable[
    [list[dict], list[int], Optional[BaseException], Optional[RecordedChunk]],
    None,
]


class SerialTranscriptionWorker:
    """Consume chunks in one worker thread and preserve submission order."""

    def __init__(
        self,
        processor: Any,
        *,
        on_progress: Optional[ProgressCallback] = None,
    ) -> None:
        self.processor = processor
        self.on_progress = on_progress
        self.segments: list[dict] = []
        self.completed_chunks: list[int] = []
        self.error: Optional[BaseException] = None
        self.failed_chunk: Optional[RecordedChunk] = None
        self._queue: queue.Queue[Optional[RecordedChunk]] = queue.Queue()
        self._thread = threading.Thread(
            target=self._run,
            name="audio-transcriber-live-worker",
            daemon=False,
        )

    def start(self) -> None:
        self._thread.start()

    def submit(self, chunk: RecordedChunk) -> None:
        if self.error is not None:
            raise LiveTranscriptionError(str(self.error))
        self._queue.put(chunk)

    def raise_if_failed(self) -> None:
        if self.error is not None:
            raise LiveTranscriptionError(str(self.error)) from self.error

    def finish(self) -> list[dict]:
        if self._thread.is_alive():
            self._queue.put(None)
            self._thread.join()
        self.raise_if_failed()
        return list(self.segments)

    def _run(self) -> None:
        current: Optional[RecordedChunk] = None
        try:
            while True:
                current = self._queue.get()
                if current is None:
                    break
                new_segments = self.processor.feed_chunk(current)
                self.segments.extend(new_segments)
                self.completed_chunks.append(current.index)
                for segment in new_segments:
                    print(
                        f"[live {segment['start_ms'] / 1000:9.1f}s] "
                        f"{segment['text']}",
                        flush=True,
                    )
                if self.on_progress is not None:
                    self.on_progress(
                        list(self.segments),
                        list(self.completed_chunks),
                        None,
                        None,
                    )
            tail = self.processor.flush()
            self.segments.extend(tail)
            for segment in tail:
                print(
                    f"[live {segment['start_ms'] / 1000:9.1f}s] "
                    f"{segment['text']}",
                    flush=True,
                )
            if self.on_progress is not None:
                self.on_progress(
                    list(self.segments),
                    list(self.completed_chunks),
                    None,
                    None,
                )
        except BaseException as exc:
            self.error = exc
            self.failed_chunk = current
            if self.on_progress is not None:
                self.on_progress(
                    list(self.segments),
                    list(self.completed_chunks),
                    exc,
                    current,
                )


def _redact(value: Any, secrets: Sequence[str]) -> Any:
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, list):
        return [_redact(item, secrets) for item in value]
    if isinstance(value, dict):
        return {key: _redact(item, secrets) for key, item in value.items()}
    return value


class AppendOnlyJournal:
    """Durable JSONL event journal for reconstructing a live session."""

    def __init__(self, path: Path, *, secrets: Sequence[str] = ()) -> None:
        self.path = Path(path)
        self.secrets = tuple(secrets)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, event: str, **details: Any) -> None:
        record = _redact(
            {
                "event": event,
                "at": datetime.now().astimezone().isoformat(timespec="seconds"),
                **details,
            },
            self.secrets,
        )
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            _fsync_parent(self.path)


def _fsync_parent(path: Path) -> None:
    try:
        directory_fd = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _checkpoint_member(path: Path, checkpoint_path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(checkpoint_path.parent.resolve()))
    except ValueError:
        return str(resolved)


def _resolve_checkpoint_member(value: str, checkpoint_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (checkpoint_path.parent / path).resolve()


def _write_live_state(
    path: Path,
    state: dict,
    *,
    secrets: Sequence[str] = (),
) -> None:
    safe_state = _redact(state, secrets)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as stream:
            json.dump(safe_state, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
        _fsync_parent(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def save_live_checkpoint(
    path: Path,
    *,
    status: str,
    recording_path: Path,
    model: str,
    base_url: str,
    audio_tag: str,
    chunk_seconds: float,
    completed_chunks: list[int],
    segments: list[dict],
    failed_at: Optional[dict] = None,
    journal_path: Optional[Path] = None,
    num_speakers: Optional[int] = None,
    speaker_names: Optional[list[str]] = None,
    output_paths: Optional[dict[str, str]] = None,
    secrets: Sequence[str] = (),
) -> None:
    """Atomically persist live progress using an explicit secret-free schema."""
    path = path.expanduser().resolve()
    normalized_outputs = None
    if output_paths:
        normalized_outputs = {
            key: _checkpoint_member(Path(value), path)
            for key, value in output_paths.items()
        }
    state = {
        "version": 1,
        "status": status,
        "backend": "api",
        "model": model,
        "base_url": base_url,
        "audio_tag": audio_tag,
        "recording_path": _checkpoint_member(recording_path, path),
        "sample_rate": SAMPLE_RATE,
        "chunk_seconds": chunk_seconds,
        "completed_chunks": completed_chunks,
        "segments": segments,
        "failed_at": failed_at,
        "journal_path": (
            _checkpoint_member(journal_path, path) if journal_path else None
        ),
        "num_speakers": num_speakers,
        "speaker_names": speaker_names,
        "output_paths": normalized_outputs,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    _write_live_state(path, state, secrets=secrets)


def capture_microphone(
    writer: PcmChunkWriter,
    worker: SerialTranscriptionWorker,
    *,
    input_device: Optional[str | int],
    duration: Optional[float],
    block_ms: int = 100,
    queue_seconds: int = 10,
    sounddevice_module: Any = None,
    on_chunk: Optional[Callable[[RecordedChunk], None]] = None,
) -> None:
    """Capture microphone PCM while keeping model/network work off callback."""
    if block_ms <= 0:
        raise ValueError("block_ms must be greater than zero")
    if duration is not None and duration <= 0:
        raise ValueError("duration must be greater than zero")
    if sounddevice_module is None:
        try:
            import sounddevice as sounddevice_module
        except ImportError as exc:
            raise LiveTranscriptionError(
                "Live recording requires sounddevice. Run scripts/setup_env.sh "
                "or install the 'live' extra."
            ) from exc

    block_frames = max(1, round(SAMPLE_RATE * block_ms / 1000))
    capacity = max(2, round(queue_seconds * 1000 / block_ms))
    audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=capacity)
    callback_errors: list[str] = []

    def callback(indata, _frames, _time_info, status) -> None:
        if status:
            callback_errors.append(f"audio input status: {status}")
            return
        try:
            audio_queue.put_nowait(bytes(indata))
        except queue.Full:
            callback_errors.append(
                "audio capture queue overflowed; recording stopped to avoid "
                "silently dropping microphone data"
            )

    started = time.monotonic()
    try:
        with sounddevice_module.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=block_frames,
            device=input_device,
            channels=CHANNELS,
            dtype="int16",
            callback=callback,
        ):
            print("Recording. Press Ctrl+C to stop.", flush=True)
            while True:
                worker.raise_if_failed()
                if callback_errors:
                    raise LiveTranscriptionError(callback_errors[0])
                if duration is not None and time.monotonic() - started >= duration:
                    break
                try:
                    pcm = audio_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                for chunk in writer.write(pcm):
                    if on_chunk is not None:
                        on_chunk(chunk)
                    worker.submit(chunk)
    except KeyboardInterrupt:
        print("\nStopping recording; draining captured audio...", flush=True)

    if callback_errors:
        raise LiveTranscriptionError(callback_errors[0])
    while True:
        try:
            pcm = audio_queue.get_nowait()
        except queue.Empty:
            break
        for chunk in writer.write(pcm):
            if on_chunk is not None:
                on_chunk(chunk)
            worker.submit(chunk)


def recording_duration_ms(path: Path) -> int:
    with wave.open(str(path), "rb") as wav:
        return round(wav.getnframes() * 1000 / wav.getframerate())


def finalize_live_output(
    recording_path: Path,
    segments: list[dict],
    *,
    raw_json_path: Path,
    markdown_path: Path,
    num_speakers: Optional[int],
    speaker_names: Optional[list[str]],
    device: str,
    model: str,
    title: str,
    assign_speakers_fn: Optional[Callable[..., list]] = None,
) -> list[dict]:
    """Run global CAM++, save normalized JSON, and render Markdown."""
    if not segments:
        raise LiveTranscriptionError(
            "No speech was recognized. The WAV recording was preserved."
        )
    assigner = assign_speakers_fn or assign_speakers_via_cam
    preset = MODEL_PRESETS["mimo"]
    assigned = assigner(
        segments,
        str(recording_path),
        num_speakers,
        preset["spk"],
        device,
    )
    return write_diarized_live_output(
        recording_path,
        assigned,
        raw_json_path=raw_json_path,
        markdown_path=markdown_path,
        speaker_names=speaker_names,
        model=model,
        title=title,
    )


def write_diarized_live_output(
    recording_path: Path,
    segments: list[dict],
    *,
    raw_json_path: Path,
    markdown_path: Path,
    speaker_names: Optional[list[str]],
    model: str,
    title: str,
) -> list[dict]:
    """Normalize an already-diarized transcript and save both output formats."""
    diarized = [
        {
            "speaker": int(segment["speaker"]),
            "start_ms": int(segment["start_ms"]),
            "end_ms": int(segment["end_ms"]),
            "text": str(segment["text"]),
        }
        for segment in segments
    ]
    raw_json_path.write_text(
        json.dumps(diarized, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    merged = merge_consecutive(diarized)
    speaker_map = build_speaker_map(diarized, speaker_names)
    cleaned_parts = [
        format_chunk(chunk, speaker_map) for chunk in chunk_by_duration(merged)
    ]
    actual_speakers = sorted({segment["speaker"] for segment in diarized})
    markdown = assemble_markdown(
        cleaned_parts,
        {
            "title": title,
            "filename": recording_path.name,
            "duration_ms": recording_duration_ms(recording_path),
            "num_speakers": len(actual_speakers),
            "language": "MiMo-V2.5-ASR (HTTP API, live recording)",
            "asr_engine": f"MiMo API ({model})",
            "speakers": [
                speaker_map.get(speaker, f"Speaker {speaker + 1}")
                for speaker in actual_speakers
            ],
            "speaker_genders": {},
        },
    )
    markdown_path.write_text(markdown, encoding="utf-8")
    return diarized


def recover_live_checkpoint(
    checkpoint_path: Path,
    *,
    api_key: str,
    device: str,
    title: str,
    api_timeout: float,
    allow_reasoning_content: bool,
    max_audio_bytes: int,
    transcribe_fn: Optional[Callable[..., list]] = None,
) -> tuple[Path, Path]:
    """Safely rebuild final output from the crash-readable master WAV.

    Recovery deliberately re-runs VAD/ASR over the complete recording. Reusing
    an arbitrary live chunk boundary could lose the processor's carried VAD
    state and produce a mixed transcript.
    """
    checkpoint_path = checkpoint_path.expanduser().resolve()
    try:
        state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise LiveTranscriptionError(
            f"Cannot read live checkpoint {checkpoint_path}: {exc}"
        ) from exc
    if not isinstance(state, dict) or state.get("backend") != "api":
        raise LiveTranscriptionError(
            "Live recovery requires an API-backend checkpoint"
        )
    required = ("recording_path", "model", "base_url", "audio_tag")
    missing = [field for field in required if not state.get(field)]
    if missing:
        raise LiveTranscriptionError(
            "Live checkpoint is missing required fields: " + ", ".join(missing)
        )
    recording_path = _resolve_checkpoint_member(
        state["recording_path"], checkpoint_path
    )
    if not recording_path.is_file():
        raise LiveTranscriptionError(
            f"Recorded WAV referenced by checkpoint does not exist: "
            f"{recording_path}"
        )
    output_paths = state.get("output_paths") or {}
    raw_json_path = _resolve_checkpoint_member(
        str(output_paths.get(
            "raw_json",
            recording_path.with_name(
                f"{recording_path.stem}_raw_transcript.json"
            ),
        )),
        checkpoint_path,
    )
    markdown_path = _resolve_checkpoint_member(
        str(output_paths.get(
            "markdown",
            recording_path.with_name(f"{recording_path.stem}-transcript.md"),
        )),
        checkpoint_path,
    )
    if transcribe_fn is None:
        from .mimo_asr import transcribe_with_mimo

        transcribe_fn = transcribe_with_mimo
    diarized = transcribe_fn(
        str(recording_path),
        num_speakers=state.get("num_speakers"),
        audio_tag=state["audio_tag"],
        device=device,
        backend="api",
        api_key=api_key,
        api_base_url=state["base_url"],
        api_model=state["model"],
        api_timeout=api_timeout,
        api_allow_reasoning_content=allow_reasoning_content,
        api_max_audio_bytes=max_audio_bytes,
    )
    write_diarized_live_output(
        recording_path,
        diarized,
        raw_json_path=raw_json_path,
        markdown_path=markdown_path,
        speaker_names=state.get("speaker_names"),
        model=state["model"],
        title=title,
    )
    state["status"] = "complete"
    state["failed_at"] = None
    state["segments"] = diarized
    state["updated_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    _write_live_state(checkpoint_path, state, secrets=(api_key,))
    journal_path = state.get("journal_path")
    if journal_path:
        AppendOnlyJournal(
            _resolve_checkpoint_member(journal_path, checkpoint_path),
            secrets=(api_key,),
        ).append(
            "session_recovered",
            checkpoint_path=str(checkpoint_path),
            segment_count=len(diarized),
        )
    return raw_json_path, markdown_path


def list_input_devices(sounddevice_module: Any = None) -> None:
    if sounddevice_module is None:
        try:
            import sounddevice as sounddevice_module
        except ImportError as exc:
            raise LiveTranscriptionError(
                "Listing microphones requires sounddevice. "
                "Run scripts/setup_env.sh or install the 'live' extra."
            ) from exc
    print(sounddevice_module.query_devices())


def _parse_input_device(value: Optional[str]) -> Optional[str | int]:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def _remove_completed_chunks(chunk_dir: Path) -> None:
    if not chunk_dir.is_dir():
        return
    for path in chunk_dir.glob("chunk_*.wav"):
        path.unlink()
    try:
        chunk_dir.rmdir()
    except OSError:
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Record chunked near-real-time microphone audio and transcribe "
            "through the Xiaomi MiMo HTTP API"
        )
    )
    parser.add_argument("--name", default=None, help="Output stem")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    parser.add_argument(
        "--input-device",
        default=None,
        help="Microphone device index or name; omit for system default",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List microphone devices and exit",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop after this many seconds; otherwise press Ctrl+C",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=15,
        help="Disk spool and recognition interval (default: 15)",
    )
    parser.add_argument(
        "--boundary-guard-ms",
        type=int,
        default=350,
        help="Carry speech this close to a chunk edge (default: 350)",
    )
    parser.add_argument(
        "--max-pending-seconds",
        type=float,
        default=60,
        help="Maximum speech carry before a forced split (default: 60)",
    )
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--speakers", default=None, help="Comma-separated names")
    parser.add_argument(
        "--device",
        default=None,
        help="VAD/CAM++ device (default: auto-detect CUDA, else cpu)",
    )
    parser.add_argument(
        "--mimo-audio-tag",
        choices=("<chinese>", "<english>", "<auto>"),
        default="<auto>",
    )
    parser.add_argument("--mimo-api-base-url", default=None)
    parser.add_argument("--mimo-api-model", default=None)
    parser.add_argument("--mimo-api-timeout", type=float, default=120)
    parser.add_argument("--mimo-api-key-env", default="MIMO_API_KEY")
    parser.add_argument(
        "--mimo-api-allow-reasoning-content",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Opt in to reasoning_content as a compatibility transcript field",
    )
    parser.add_argument("--mimo-api-max-audio-mb", type=float, default=None)
    parser.add_argument(
        "--recover-checkpoint",
        type=Path,
        default=None,
        help="Rebuild output from a preserved *_live_partial.json and WAV; "
             "does not access the microphone",
    )
    parser.add_argument("--title", default="Live Transcript")
    parser.add_argument(
        "--keep-chunks",
        action="store_true",
        help="Keep intermediate spool WAV files after success",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output files with the selected name",
    )
    return parser


def run(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_devices:
        list_input_devices()
        return 0
    if args.chunk_seconds < 2 or args.chunk_seconds > 120:
        parser.error("--chunk-seconds must be between 2 and 120")
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be greater than zero")
    if args.boundary_guard_ms < 0:
        parser.error("--boundary-guard-ms must not be negative")
    if args.max_pending_seconds <= 0:
        parser.error("--max-pending-seconds must be greater than zero")

    api = resolve_mimo_api_config(
        args.mimo_api_base_url,
        args.mimo_api_model,
        args.mimo_api_timeout,
        args.mimo_api_key_env,
        args.mimo_api_allow_reasoning_content,
        args.mimo_api_max_audio_mb,
    )
    if not api["api_key"]:
        parser.error(
            f"MiMo API key is not set in environment variable "
            f"{args.mimo_api_key_env}"
        )

    if args.recover_checkpoint is not None:
        if args.device is None:
            try:
                import torch

                args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                args.device = "cpu"
        try:
            raw_path, markdown_path = recover_live_checkpoint(
                args.recover_checkpoint.expanduser(),
                api_key=api["api_key"],
                device=args.device,
                title=args.title,
                api_timeout=api["timeout"],
                allow_reasoning_content=api["allow_reasoning_content"],
                max_audio_bytes=api["max_audio_bytes"],
            )
        except Exception as exc:
            safe_error = _safe_error_text(exc, (api["api_key"],))
            print(f"Error: live recovery failed: {safe_error}", file=sys.stderr)
            return 1
        print(f"Recovered raw transcript: {raw_path}")
        print(f"Recovered Markdown: {markdown_path}")
        return 0

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or datetime.now().strftime("live-%Y%m%d-%H%M%S")
    if Path(name).name != name or name in ("", ".", ".."):
        parser.error("--name must be a single safe filename stem")
    recording_path = output_dir / f"{name}.wav"
    raw_json_path = output_dir / f"{name}_raw_transcript.json"
    markdown_path = output_dir / f"{name}-transcript.md"
    checkpoint_path = output_dir / f"{name}_live_partial.json"
    journal_path = output_dir / f"{name}_live_journal.jsonl"
    chunk_dir = output_dir / f"{name}_live_chunks"
    targets = (
        recording_path,
        raw_json_path,
        markdown_path,
        checkpoint_path,
        journal_path,
    )
    existing = [path for path in targets if path.exists()]
    if chunk_dir.exists() and any(chunk_dir.iterdir()):
        existing.append(chunk_dir)
    if existing and not args.overwrite:
        parser.error(
            "output already exists; choose another --name or pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    if args.overwrite:
        for path in targets:
            if path.is_file():
                path.unlink()
        _remove_completed_chunks(chunk_dir)

    if args.device is None:
        try:
            import torch

            args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"
    speaker_names = (
        [name.strip() for name in args.speakers.split(",") if name.strip()]
        if args.speakers
        else None
    )
    num_speakers = args.num_speakers or (
        len(speaker_names) if speaker_names else None
    )
    if num_speakers is not None and num_speakers < 1:
        parser.error("--num-speakers must be at least 1")

    print(f"Compute device: {args.device}")
    print(f"Recording: {recording_path}")
    print(f"MiMo API: {api['base_url']} ({api['model']})")
    print("Loading FSMN VAD before microphone capture...")
    try:
        from funasr import AutoModel

        vad_model = AutoModel(
            model=MODEL_PRESETS["mimo"]["vad"],
            model_revision=modelscope_revision(MODEL_PRESETS["mimo"]["vad"]),
            vad_kwargs={"max_single_segment_time": 30_000},
            device=args.device,
            disable_update=True,
        )
        recognizer = MimoApiRecognizer(
            api_key=api["api_key"],
            base_url=api["base_url"],
            model=api["model"],
            timeout=api["timeout"],
            allow_reasoning_content=api["allow_reasoning_content"],
            max_audio_bytes=api["max_audio_bytes"],
        )
        processor = StreamingMimoApiProcessor(
            MimoApiEngine(recognizer),
            vad_model,
            audio_tag=args.mimo_audio_tag,
            api_key=api["api_key"],
            boundary_guard_ms=args.boundary_guard_ms,
            max_pending_seconds=args.max_pending_seconds,
        )
    except Exception as exc:
        print(f"Error: failed to initialize live transcription: {exc}", file=sys.stderr)
        return 1

    journal = AppendOnlyJournal(journal_path, secrets=(api["api_key"],))
    output_paths = {
        "raw_json": str(raw_json_path),
        "markdown": str(markdown_path),
    }
    journal.append(
        "session_started",
        recording_path=str(recording_path),
        checkpoint_path=str(checkpoint_path),
        model=api["model"],
        base_url=api["base_url"],
        audio_tag=args.mimo_audio_tag,
        chunk_seconds=args.chunk_seconds,
    )

    def checkpoint_progress(
        segments: list[dict],
        completed: list[int],
        error: Optional[BaseException],
        failed_chunk: Optional[RecordedChunk],
    ) -> None:
        failed_at = None
        status = "recording"
        if error is not None:
            status = "failed"
            failed_at = {
                "chunk": failed_chunk.index if failed_chunk else None,
                "start_ms": failed_chunk.start_ms if failed_chunk else None,
                "end_ms": failed_chunk.end_ms if failed_chunk else None,
                "error": _safe_error_text(error, (api["api_key"],)),
            }
        save_live_checkpoint(
            checkpoint_path,
            status=status,
            recording_path=recording_path,
            model=api["model"],
            base_url=api["base_url"],
            audio_tag=args.mimo_audio_tag,
            chunk_seconds=args.chunk_seconds,
            completed_chunks=completed,
            segments=segments,
            failed_at=failed_at,
            journal_path=journal_path,
            num_speakers=num_speakers,
            speaker_names=speaker_names,
            output_paths=output_paths,
            secrets=(api["api_key"],),
        )
        journal.append(
            "progress" if error is None else "recognition_failed",
            completed_chunks=completed,
            segment_count=len(segments),
            failed_at=failed_at,
        )

    writer = PcmChunkWriter(
        recording_path,
        chunk_dir,
        chunk_seconds=args.chunk_seconds,
    )
    worker = SerialTranscriptionWorker(
        processor,
        on_progress=checkpoint_progress,
    )
    worker.start()
    save_live_checkpoint(
        checkpoint_path,
        status="recording",
        recording_path=recording_path,
        model=api["model"],
        base_url=api["base_url"],
        audio_tag=args.mimo_audio_tag,
        chunk_seconds=args.chunk_seconds,
        completed_chunks=[],
        segments=[],
        journal_path=journal_path,
        num_speakers=num_speakers,
        speaker_names=speaker_names,
        output_paths=output_paths,
        secrets=(api["api_key"],),
    )

    try:
        capture_microphone(
            writer,
            worker,
            input_device=_parse_input_device(args.input_device),
            duration=args.duration,
            on_chunk=lambda chunk: journal.append(
                "chunk_committed",
                index=chunk.index,
                path=chunk.path,
                start_ms=chunk.start_ms,
                end_ms=chunk.end_ms,
            ),
        )
        for chunk in writer.close():
            journal.append(
                "chunk_committed",
                index=chunk.index,
                path=chunk.path,
                start_ms=chunk.start_ms,
                end_ms=chunk.end_ms,
            )
            worker.submit(chunk)
        segments = worker.finish()
        print("Recording complete. Running global CAM++ speaker clustering...")
        diarized = finalize_live_output(
            recording_path,
            segments,
            raw_json_path=raw_json_path,
            markdown_path=markdown_path,
            num_speakers=num_speakers,
            speaker_names=speaker_names,
            device=args.device,
            model=api["model"],
            title=args.title,
        )
        save_live_checkpoint(
            checkpoint_path,
            status="complete",
            recording_path=recording_path,
            model=api["model"],
            base_url=api["base_url"],
            audio_tag=args.mimo_audio_tag,
            chunk_seconds=args.chunk_seconds,
            completed_chunks=worker.completed_chunks,
            segments=diarized,
            journal_path=journal_path,
            num_speakers=num_speakers,
            speaker_names=speaker_names,
            output_paths=output_paths,
            secrets=(api["api_key"],),
        )
        journal.append(
            "session_complete",
            segment_count=len(diarized),
            raw_json_path=str(raw_json_path),
            markdown_path=str(markdown_path),
        )
        if not args.keep_chunks:
            _remove_completed_chunks(chunk_dir)
        print(f"Raw transcript: {raw_json_path}")
        print(f"Markdown: {markdown_path}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Journal: {journal_path}")
        return 0
    except Exception as exc:
        writer.close()
        try:
            worker.finish()
        except Exception:
            pass
        safe_error = _safe_error_text(exc, (api["api_key"],))
        checkpoint_progress(
            list(worker.segments),
            list(worker.completed_chunks),
            exc,
            worker.failed_chunk,
        )
        print(f"Error: {safe_error}", file=sys.stderr)
        print(
            f"Recording and live checkpoint were preserved: "
            f"{recording_path}, {checkpoint_path}",
            file=sys.stderr,
        )
        return 1
    finally:
        processor.close()


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
