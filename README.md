# Audio Transcriber

Standalone Python CLI for transcribing meetings and podcasts with speaker
diarization, resumable processing, and optional LLM cleanup.

Two ASR engine families are available:

- **[FunASR](https://github.com/modelscope/FunASR)** — Paraformer,
  SenseVoice, and Whisper presets; supports CPU and GPU.
- **[MiMo-V2.5-ASR](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-ASR)** —
  either a local 8B model or Xiaomi's HTTP API.

Both paths use the same local FSMN VAD and CAM++ speaker pipeline and produce
the same `speaker` / `start_ms` / `end_ms` / `text` segment format.

## Features

- Chinese, English, automatic language detection, and Whisper presets.
- CAM++ speaker embeddings and clustering with real-name mapping.
- Hotword biasing on the Chinese SeACo-Paraformer preset.
- Checkpoints for interrupted MiMo recognition and LLM cleanup.
- Optional cleanup through Amazon Bedrock, Anthropic, or an
  OpenAI-compatible API.
- Markdown and raw JSON output.
- CPU support for FunASR and MiMo API workflows.

## Installation

The setup script creates `.venv`, installs the local package, and installs the
FunASR/CAM++ runtime used by transcription, VAD, and speaker clustering. It
requires Python 3.12:

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
```

MiMo API mode needs no MiMo weights. Local MiMo is an explicit opt-in because
it downloads approximately 34 GB and requires a CUDA GPU with at least 20 GB
VRAM:

```bash
INSTALL_MIMO=1 MIMO_WEIGHTS_PATH=/path/to/hf-cache \
  bash scripts/setup_env.sh
```

For development and mocked tests only:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
python -m pytest -q
```

`ffmpeg` and `ffprobe` must be available on `PATH` for input conversion.

## Quick start

```bash
# Chinese meeting with speaker count and hotwords
audio-transcriber meeting.m4a \
  --lang zh \
  --num-speakers 4 \
  --hotwords "张三 李四 产品代号"

# English meeting with output names
audio-transcriber meeting.wav \
  --lang en \
  --speakers "Alice,Bob,Carol"

# Automatic language detection
audio-transcriber interview.mp3 \
  --lang auto \
  --num-speakers 2

# Raw ASR and diarization only, without LLM cleanup
audio-transcriber meeting.wav \
  --lang zh \
  --skip-llm
```

The default outputs are `<stem>_raw_transcript.json` and
`<stem>-transcript.md`. Pass `--json-out` or `--output` to override them.

## MiMo local and API backends

`--lang mimo` keeps FSMN VAD, CAM++, checkpoints, post-processing, and output
local. `--mimo-backend` changes only the per-segment recognizer:

| Backend | Requirements | ASR Engine metadata |
|---|---|---|
| `local` (default) | MiMo weights, CUDA, at least 20 GB VRAM | `MiMo-V2.5-ASR (local)` |
| `api` | Xiaomi MiMo API key; VAD/CAM++ may run on CPU | `MiMo API (mimo-v2.5-asr)` |

### MiMo API

```bash
export MIMO_API_KEY='...'

audio-transcriber meeting.m4a \
  --lang mimo \
  --mimo-backend api \
  --mimo-audio-tag '<auto>' \
  --device cpu \
  --num-speakers 4 \
  --speakers '张三,李四,王五,赵六'
```

API configuration follows CLI-over-environment precedence:

| CLI option | Environment variable | Default |
|---|---|---|
| `--mimo-api-base-url` | `MIMO_BASE_URL` | `https://api.xiaomimimo.com/v1` |
| `--mimo-api-model` | `MIMO_API_MODEL` | `mimo-v2.5-asr` |
| `--mimo-api-key-env NAME` | Reads `NAME` | `MIMO_API_KEY` |
| `--mimo-api-timeout` | — | `120` seconds |

The client sends serial `input_audio` requests to
`POST {base_url}/chat/completions`. Transient HTTP 408/429/5xx responses,
connection failures, and timeouts use finite exponential backoff. A 400
request error or 401/403 authentication error fails immediately.

After an interrupted run, use the same backend, model, base URL, and audio tag:

```bash
audio-transcriber meeting.m4a \
  --lang mimo \
  --mimo-backend api \
  --mimo-audio-tag '<auto>' \
  --device cpu \
  --resume-mimo
```

The API key is read only from the selected environment variable. It is never
written to logs, exceptions, checkpoints, raw transcript JSON, or Markdown.
Audio segments are sent to the configured endpoint; local VAD and speaker
embeddings are not uploaded. `--mimo-batch` remains accepted for command-line
compatibility but is deprecated and does not enable concurrency.

### MiMo local

```bash
audio-transcriber episode.flac \
  --lang mimo \
  --mimo-backend local \
  --mimo-audio-tag '<chinese>' \
  --mimo-weights-path /path/to/hf-cache \
  --num-speakers 2
```

Local mode preserves the original MiMo behavior: it verifies CUDA and local
weights, loads the model once, recognizes VAD segments in order, and frees the
model before CAM++ clustering.

## LLM cleanup

LLM cleanup is opt-in through `--model`. Provider selection can be automatic
or explicit with `--provider bedrock|anthropic|openai`.

```bash
# Anthropic
export ANTHROPIC_API_KEY='...'
audio-transcriber meeting.wav \
  --lang zh \
  --model claude-sonnet-4-6 \
  --provider anthropic

# OpenAI-compatible endpoint
export OPENAI_API_KEY='...'
export OPENAI_BASE_URL='https://example.com/v1'
audio-transcriber meeting.wav \
  --lang zh \
  --model your-model \
  --provider openai
```

Amazon Bedrock uses the normal AWS credential chain and can be selected with
`--provider bedrock --bedrock-region <region>`. Provider credentials are not
stored in transcript artifacts.

## Speaker verification

The optional verifier analyzes existing raw JSON and can repair mislabeled
speaker IDs:

```bash
audio-transcriber-verify-speakers meeting_raw_transcript.json \
  --speakers "Alice,Bob,Carol" \
  --speaker-context speaker-context.json \
  --fix
```

## Pipeline

```text
Input audio
  -> ffmpeg: 16 kHz mono
  -> FSMN VAD
  -> FunASR or MiMo (local/API), serial by segment
  -> CAM++ embeddings and speaker clustering
  -> normalized raw JSON
  -> merge and speaker naming
  -> optional LLM cleanup
  -> Markdown
```

See [docs/pipeline-details.md](docs/pipeline-details.md) for model presets,
long-recording behavior, CPU guidance, checkpoints, and output details.

## Project layout

```text
audio_transcriber/
  transcribe.py           main pipeline and CLI
  mimo_asr.py             MiMo orchestration, retry, and checkpoints
  mimo_api.py             Xiaomi MiMo HTTP client
  llm_utils.py            Bedrock, Anthropic, and OpenAI-compatible clients
  speaker_gender.py       optional gender hints
  verify_speakers.py      speaker-label verification CLI
scripts/
  setup_env.sh            base environment installer
  setup_mimo.sh           opt-in local MiMo installer
tests/                    network-free automated tests
docs/                     detailed pipeline and historical design notes
```

## Performance

On a 4h14m, nine-speaker Chinese meeting, the FunASR path completed
transcription in approximately 169 seconds on an L40S. CPU speed varies
substantially by model and machine. The included clustering patch replaces a
cubic eigenvalue decomposition that can otherwise dominate long recordings.

For the local MiMo/FunASR comparison, see
[docs/superpowers/reports/2026-04-30-mimo-vs-funasr-perf-cost.md](docs/superpowers/reports/2026-04-30-mimo-vs-funasr-perf-cost.md).

## License

MIT
