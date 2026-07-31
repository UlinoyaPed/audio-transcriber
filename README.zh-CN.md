# Audio Transcriber

[English](README.md) | [简体中文](README.zh-CN.md)

一个面向会议和播客的独立多引擎 Python 命令行工具，支持实时录音、说话人分离、可恢复的 MiMo 本地/API 后端和可选的 LLM 文本清理。

项目提供两类 ASR 引擎：

- **[FunASR](https://github.com/modelscope/FunASR)**：预置 Paraformer、SenseVoice 和 Whisper，可在 CPU 或 GPU 上运行。
- **[MiMo-V2.5-ASR](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-ASR)**：可使用本地 8B 模型，也可调用小米 MiMo HTTP API。

两条路径共用本地 FSMN VAD 和 CAM++ 说话人处理管线，最终均生成包含 `speaker`、`start_ms`、`end_ms` 和 `text` 的统一分段结构。

## 功能

- 支持中文、英文、自动语言检测和 Whisper 多语言预设。
- 使用 CAM++ 提取说话人嵌入并聚类，支持映射真实姓名。
- 中文 SeACo-Paraformer 预设支持热词偏置。
- MiMo 识别和 LLM 清理支持中断恢复。
- 支持麦克风实时录音，并通过 MiMo API 串行增量转录。
- 可通过 Amazon Bedrock、Anthropic 或 OpenAI 兼容 API 清理文本。
- 输出 Markdown 和原始 JSON。
- FunASR 与 MiMo API 工作流均可使用 CPU。

## 安装

安装脚本会创建 `.venv`、安装当前项目，并配置转录、VAD 和说话人聚类所需的 FunASR/CAM++ 运行环境。脚本要求 Python 3.12：

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
```

MiMo API 模式不需要下载 MiMo 权重。本地 MiMo 需要下载约 34 GB 数据，并要求至少具有 20 GB 显存的 CUDA GPU，因此必须显式启用：

```bash
INSTALL_MIMO=1 MIMO_WEIGHTS_PATH=/path/to/hf-cache \
  bash scripts/setup_env.sh
```

如果只进行开发和模拟测试：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
python -m pytest -q
```

音频转换要求 `PATH` 中存在 `ffmpeg` 和 `ffprobe`。实时录音还依赖 PortAudio；Debian/Ubuntu 对应 `libportaudio2`，Homebrew 对应 `portaudio`。安装脚本会安装 Python 的 `sounddevice` 绑定。

## 快速开始

```bash
# 中文会议：指定说话人数和热词
audio-transcriber meeting.m4a \
  --lang zh \
  --num-speakers 4 \
  --hotwords "张三 李四 产品代号"

# 英文会议：指定输出姓名
audio-transcriber meeting.wav \
  --lang en \
  --speakers "Alice,Bob,Carol"

# 自动检测语言
audio-transcriber interview.mp3 \
  --lang auto \
  --num-speakers 2

# 只执行 ASR 和说话人分离，不调用 LLM
audio-transcriber meeting.wav \
  --lang zh \
  --skip-llm
```

默认输出为音频旁边的 `<文件名>_raw_transcript.json` 和
`<文件名>-transcript.md`。原始 JSON 会保存源音频绝对路径、大小、纳秒级
mtime、SHA-256、ASR 参数和标准化分段。只有显式传入 `--overwrite` 才会替换
已有输出；`--skip-transcribe` 会拒绝源文件或处理参数不匹配的 raw artifact。
可分别通过 `--json-out` 和 `--output` 指定其他路径。

## 分块式近实时录音与转录

先列出麦克风设备，再开始录制：

```bash
audio-transcriber-live --list-devices

export MIMO_API_KEY='...'
audio-transcriber-live \
  --name weekly-meeting \
  --input-device 0 \
  --chunk-seconds 15 \
  --mimo-audio-tag '<auto>' \
  --device cpu \
  --num-speakers 4 \
  --speakers '张三,李四,王五,赵六'
```

按 `Ctrl+C` 停止，也可以用 `--duration 1800` 设定录制时长。每个稳定的 VAD 语音段识别完成后，终端会立即显示文本。录音数据会先写入 `weekly-meeting.wav`，随后由单一工作线程调用 API，因此网络延迟不会阻塞麦克风回调。触及定时分块边界的语音会被保留到下一窗口，降低断词概率。

实时预览暂不显示说话人姓名。停止录制后，程序会针对完整 WAV 统一运行一次 CAM++，再将稳定的说话人编号写入 `weekly-meeting_raw_transcript.json` 和 `weekly-meeting-transcript.md`。进度保存在 `weekly-meeting_live_partial.json`，持久事件追加到 `weekly-meeting_live_journal.jsonl`；API Key 会从这两类文件、日志和输出中脱敏。

失败后无需访问麦克风即可恢复：

```bash
audio-transcriber-live \
  --recover-checkpoint weekly-meeting_live_partial.json \
  --device cpu
```

恢复会重新处理可在崩溃后读取的完整主 WAV，以保证 VAD 边界正确。成功完成后，中间 WAV 分块会自动删除，使用 `--keep-chunks` 可保留。

该功能属于分块式近实时转录，正常延迟约为分块时长、VAD 耗时和 API 耗时之和。当前实时命令仅支持 MiMo HTTP API；本地 MiMo 和 FunASR 继续通过离线 `audio-transcriber` 命令使用。可靠性设计、恢复方法和限制详见[实时转录中文文档](docs/live-transcription.zh-CN.md)。

## MiMo 本地与 API 后端

使用 `--lang mimo` 时，FSMN VAD、CAM++、断点文件、后处理和输出均保留在本地。`--mimo-backend` 只切换逐段识别后端：

| 后端 | 要求 | ASR Engine 元数据 |
|---|---|---|
| `local`（默认） | MiMo 权重、CUDA、至少 20 GB 显存 | `MiMo-V2.5-ASR (local)` |
| `api` | 小米 MiMo API Key；VAD/CAM++ 可在 CPU 上运行 | `MiMo API (mimo-v2.5-asr)` |

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

API 配置遵循“命令行优先于环境变量”的规则：

| 命令行参数 | 环境变量 | 默认值 |
|---|---|---|
| `--mimo-api-base-url` | `MIMO_BASE_URL` | `https://api.xiaomimimo.com/v1` |
| `--mimo-api-model` | `MIMO_API_MODEL` | `mimo-v2.5-asr` |
| `--mimo-api-key-env NAME` | 读取 `NAME` | `MIMO_API_KEY` |
| `--mimo-api-timeout` | — | `120` 秒 |
| `--mimo-api-max-audio-mb` | `MIMO_API_MAX_AUDIO_MB` | `20` |
| `--mimo-api-allow-reasoning-content` | `MIMO_API_ALLOW_REASONING_CONTENT` | 关闭 |

客户端会按顺序向 `POST {base_url}/chat/completions` 发送 `input_audio` 请求。HTTP 408、429、5xx、网络连接失败和超时会触发有限次数的指数退避；HTTP 400 请求错误和 401/403 鉴权错误会立即失败。

默认只接受 `message.content` 作为转录文本；只有显式启用兼容开关时才回退到 `reasoning_content`。WAV 会按块进行 Base64 编码，并受上传大小限制保护。协议要求的 Base64 仍会增加约三分之一传输体积。

中断后必须使用相同的后端、模型、Base URL 和音频标签恢复：

```bash
audio-transcriber meeting.m4a \
  --lang mimo \
  --mimo-backend api \
  --mimo-audio-tag '<auto>' \
  --device cpu \
  --resume-mimo
```

API Key 只从指定环境变量读取，不会写入日志、异常信息、断点文件、原始 JSON 或 Markdown。只有音频语音段会发送到配置的端点，本地 VAD 数据和说话人嵌入不会上传。`--mimo-batch` 仅为兼容旧命令而保留，已弃用且不会启用并发。

### MiMo 本地模式

```bash
audio-transcriber episode.flac \
  --lang mimo \
  --mimo-backend local \
  --mimo-audio-tag '<chinese>' \
  --mimo-weights-path /path/to/hf-cache \
  --num-speakers 2
```

本地模式保持原有行为：检查 CUDA 和本地权重，只加载一次 MiMo，按 VAD 顺序逐段识别，在 CAM++ 聚类前释放 MiMo 模型。

## LLM 文本清理

LLM 清理通过 `--model` 显式启用。可以自动识别提供方，也可使用 `--provider bedrock|anthropic|openai` 指定。

```bash
# Anthropic
export ANTHROPIC_API_KEY='...'
audio-transcriber meeting.wav \
  --lang zh \
  --model claude-sonnet-4-6 \
  --provider anthropic

# OpenAI 兼容端点
export OPENAI_API_KEY='...'
export OPENAI_BASE_URL='https://example.com/v1'
audio-transcriber meeting.wav \
  --lang zh \
  --model your-model \
  --provider openai
```

Amazon Bedrock 使用标准 AWS 凭证链，可通过 `--provider bedrock --bedrock-region <区域>` 选择。提供方凭证不会写入转录文件。

## 说话人校验

可选校验工具能够分析已有原始 JSON，并修复错误的说话人编号：

```bash
audio-transcriber-verify-speakers meeting_raw_transcript.json \
  --speakers "Alice,Bob,Carol" \
  --speaker-context speaker-context.json \
  --fix
```

三人及以上场景只接受覆盖全部当前姓名的完整排列。`--fix` 会先生成不覆盖
已有备份的 `.bak` 文件，再对临时 JSON 执行 fsync 和解析回读，最后原子替换
目标文件。

## 处理管线

```text
输入音频
  -> ffmpeg：16 kHz 单声道
  -> FSMN VAD
  -> FunASR 或 MiMo（本地/API），按语音段串行识别
  -> CAM++ 嵌入和说话人聚类
  -> 规范化原始 JSON
  -> 合并分段并映射说话人姓名
  -> 可选 LLM 清理
  -> Markdown
```

模型预设、长录音处理、CPU 使用、断点机制和输出细节参见[管线技术细节中文文档](docs/pipeline-details.zh-CN.md)。

## 项目结构

```text
audio_transcriber/
  live.py                 麦克风录制和串行实时工作线程
  transcribe.py           主处理管线和 CLI
  mimo_asr.py             MiMo 编排、重试和断点管理
  mimo_api.py             小米 MiMo HTTP 客户端
  llm_utils.py            Bedrock、Anthropic 和 OpenAI 兼容客户端
  speaker_gender.py       可选的说话人性别提示
  verify_speakers.py      说话人标签校验 CLI
scripts/
  setup_env.sh            基础环境安装脚本
  setup_mimo.sh           可选的本地 MiMo 安装脚本
tests/                    不访问真实网络的自动化测试
docs/                     管线技术文档和历史设计资料
```

## 性能

在一段 4 小时 14 分、包含 9 名说话人的中文会议上，FunASR 路径使用 L40S 完成转录约需 169 秒。CPU 速度会因模型和机器配置产生很大差异。项目自带的聚类补丁替换了三次复杂度的特征值分解，避免其在长录音上成为主要瓶颈。

本地 MiMo 与 FunASR 的完整对比见 [`docs/superpowers/reports/2026-04-30-mimo-vs-funasr-perf-cost.md`](docs/superpowers/reports/2026-04-30-mimo-vs-funasr-perf-cost.md)。

## 许可证

MIT
