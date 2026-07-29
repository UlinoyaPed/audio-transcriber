# 音频转录管线——技术细节

[English](pipeline-details.md) | [简体中文](pipeline-details.zh-CN.md)

麦克风采集和 MiMo API 增量识别详见[实时录音与转录](live-transcription.zh-CN.md)。实时模式与离线模式使用相同的最终 CAM++、JSON 和 Markdown 结构，但会将全局说话人聚类延后到录制结束。

## 架构

```text
音频文件（.m4a/.mp3/.wav）
  │
  ├─ [ffmpeg] ──► 16 kHz 单声道 WAV
  │
  ├─ [阶段 1：ASR] ──► raw_transcript.json
  │   ├─ FSMN-VAD：区分语音与静音
  │   ├─ ASR 模型：根据语言预设选择
  │   ├─ 可选热词偏置：仅 SeACo-Paraformer
  │   ├─ 标点恢复：取决于模型
  │   └─ CAM++：说话人嵌入与聚类
  │
  ├─ [阶段 2：后处理]
  │   ├─ 合并间隔小于 2 秒的同一说话人分段
  │   ├─ 将说话人编号映射为姓名
  │   └─ 通过自我介绍自动校验姓名映射
  │
  └─ [阶段 3：LLM 清理] ──► transcript.md
      ├─ 使用 --speaker-context 校验说话人角色
      ├─ 删除“嗯、啊、um、uh”等填充词
      ├─ 根据上下文修复同音词等 ASR 错误
      ├─ 在不改变原意的前提下整理语法
      └─ 可根据上下文区分被错误合并的说话人
```

## 语言预设与模型

### `--lang zh`：中文默认预设，SeACo-Paraformer，支持热词

| 组件 | 模型 ID | 参数量 | 用途 |
|---|---|---:|---|
| ASR | `iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` | 220M | 中文 ASR，CER 1.95%，支持热词定制 |
| VAD | `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` | 0.4M | 语音活动检测 |
| 标点 | `iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch` | 290M | 标点恢复 |
| 说话人 | `iic/speech_campplus_sv_zh-cn_16k-common` | 7.2M | 说话人分离 |

SeACo-Paraformer 接受 `--hotwords`，值可以是空格分隔的字符串，也可以是 `.txt` 文件，用于提升指定术语的识别概率。

### `--lang zh-basic`：中文，不使用热词

与 `zh` 基本相同，但使用不支持热词的基础 Paraformer-large。无需热词，或者热词影响英文术语时，可以使用该预设。

| 组件 | 模型 ID | 参数量 | 用途 |
|---|---|---:|---|
| ASR | `iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch` | 220M | 中文 ASR，CER 1.95% |

### `--lang en`：英文

| 组件 | 模型 ID | 参数量 | 用途 |
|---|---|---:|---|
| ASR | `iic/speech_paraformer-large-vad-punc_asr_nat-en-16k-common-vocab10020` | 220M | 英文 ASR |

### `--lang auto`：自动检测中文、英文、日文、韩文和粤语

| 组件 | 模型 ID | 参数量 | 用途 |
|---|---|---:|---|
| ASR | `iic/SenseVoiceSmall` | 234M | 自动检测语言的多语言 ASR |

SenseVoiceSmall 内置标点，并支持情绪检测。

### `--lang whisper`：99 种语言

| 组件 | 模型 ID | 参数量 | 用途 |
|---|---|---:|---|
| ASR | `iic/Whisper-large-v3-turbo` | 809M | 通过 FunASR 运行 OpenAI Whisper，覆盖语言最多 |

所有 FunASR 预设共用 FSMN VAD 和 CAM++。模型会在首次运行时从 ModelScope 自动下载。

## 热词偏置

SeACo-Paraformer（`--lang zh`）可以提升特定词语的识别概率，适合会议中的参会者姓名、项目名、产品名和行业术语。

### 提供热词

```bash
# 空格分隔
audio-transcriber meeting.wav --lang zh --hotwords "张三 李四 ClawCon Rebase"

# 文本文件，每行一个词
audio-transcriber meeting.wav --lang zh --hotwords hotwords.txt
```

会议热词文件通常应包含：

- 参会者全名；
- 项目名、内部代号和产品品牌；
- 中文行业术语和中文缩略词；
- 公司、团队和部门名称。

### 实测效果

在一段 4 小时 14 分、9 名说话人的中文会议上使用 27 个热词，结果如下：

| 术语 | 不使用热词 | 使用热词 | 变化 |
|---|---:|---:|---|
| 龙虾（lobechat） | 28 | 42 | **+50%** |
| 高琦（姓名） | 0 | 7 | **0 → 7** |
| 搬瓦工（BandwagonHost） | 0 | 1 | **0 → 1** |
| 谢锐（姓名） | 0 | 1 | 有改善 |
| 鲲鹏（组织名） | 6 | 7 | 略有改善 |
| Rebase（英文） | 5 | 0 | **退化** |
| Tailwind（英文） | 3 | 1 | **退化** |

主要结论：

- 中文姓名、品牌和术语通常有明显改善；
- SeACo 热词偏置基于中文词元，英文借词可能受到干扰；
- 参会者很少在会议中完整念出他人姓名，因此姓名热词只在实际出现时有效。

建议把中文术语和姓名加入热词。英文技术词优先交给阶段 3 的 LLM 清理；如果英文准确率很重要，而且热词造成退化，可改用 `--lang zh-basic`。

## 性能基准

测试音频为 4 小时 14 分、9 名说话人的中文会议，GPU 为 L40S 46 GB：

| 指标 | Paraformer-large | SeACo + 热词 | 说明 |
|---|---:|---:|---|
| 模型加载 | 14 秒 | 14 秒；首次运行下载 944 MB 模型需 422 秒 | 后续使用缓存 |
| 转录 | 169 秒 | 168 秒 | 基本相同 |
| 原始句子 | 6672 | 6725 | 接近 |
| 合并后分段 | 1695 | 1724 | 接近 |
| 检出的说话人 | 7/9 | 7/9 | 聚类结果相同 |

| 阶段 | GPU（L40S 46 GB） | CPU 估计值 |
|---|---:|---:|
| 模型加载 | 14 秒 | 约 30 秒 |
| 转录 | 169 秒 | 约 30–60 分钟 |
| 说话人聚类 | 约 10 秒，已打补丁 | 约 2–5 分钟，已打补丁 |
| LLM 清理，17 个分块 | 约 35 分钟 | 约 35 分钟，主要受网络限制 |
| 总计 | 约 38 分钟 | 约 70–100 分钟 |

未应用聚类补丁时，原始 `scipy.linalg.eigh()` 会在完整拉普拉斯矩阵上执行 $O(N^3)$ 计算，该录音需要 10 小时以上。补丁使用 `scipy.sparse.linalg.eigsh()`，将复杂度降至约 $O(N^2k)$。

## 长会议聚类补丁

FunASR 的 `SpectralCluster.get_spec_embs()` 使用 `scipy.linalg.eigh(L)` 计算 $N \times N$ 拉普拉斯矩阵的全部特征值。4 小时录音可能产生 6000 个以上分段，使三次复杂度计算持续数小时。

`audio-transcriber-patch-clustering` 会进行两项修改：

- 使用 `scipy.sparse.linalg.eigsh(L_sparse, k=num_speakers, which='SM')`，只计算所需的最小 $k$ 个特征值；
- 使用 NumPy 广播向量化 `p_pruning()`，替换 Python 循环。

处理约 1 小时以上的会议前应先应用补丁。`scripts/setup_env.sh` 会自动执行。

## 说话人角色校验

说话人姓名默认按照首次出现顺序分配，这在播客中可能交换主持人与嘉宾。项目提供三层校验：

- **阶段 2，自我介绍检测**：扫描前 5 分钟，识别“我是 X”“I'm X”等明确自我介绍，必要时交换标签；
- **阶段 3，LLM 角色校验**：提供 `--speaker-context` 后，LLM 会在文本清理前分析第一个最长 15 分钟的分块。两人场景执行 CORRECT/SWAP 判断，三人以上使用 JSON 完整重映射；
- **事后校验工具**：`audio-transcriber-verify-speakers` 可检查任意 `*_raw_transcript.json`，支持预览两人交换或多人重映射，并可用 `--fix` 写回。

播客场景建议始终提供描述主持人和嘉宾角色的 `--speaker-context`。

## 说话人分离限制

CAM++ 可能把声学特征相似的人合并为同一编号。在上述 9 人会议中只识别出 7 个唯一编号，其中两对被合并。

可采用以下办法：

- 使用 `--num-speakers N` 给出预期人数；
- 根据议程、参会记录等参考资料，在转录后通过关键词映射姓名；
- 使用 `--speaker-context` 提供每个人的关键词，让 LLM 在上下文足够明确时拆分被合并的说话人，实测成功率约 73%。

## 提升效果所需的辅助文件

| 文件 | 使用阶段 | 用途 |
|---|---|---|
| `hotwords.txt` | 阶段 1，`--hotwords` | 提升姓名和术语识别概率 |
| `speaker-context.json` | 阶段 3，`--speaker-context` | 帮助 LLM 识别和拆分说话人 |
| 会议议程 | 人工参考 | 识别会议阶段 |
| 参会者名单 | 构建热词与姓名参数 | 将编号映射为真实姓名 |

根据会议邀请准备辅助文件：

```bash
# 1. 根据参会者和议程创建 hotwords.txt
cat > hotwords.txt << 'EOF'
Alice
Bob
Carol
ProjectAlpha
Sprint Review
Q2 OKR
EOF

# 2. 根据参会者角色创建 speaker-context.json
cat > speaker-context.json << 'EOF'
{
  "Alice": "Engineering manager, discusses sprint velocity and tech debt",
  "Bob": "Product manager, presents roadmap and customer feedback",
  "Carol": "Designer, shows mockups, mentions Figma and user testing"
}
EOF

# 3. 同时使用两个文件
audio-transcriber meeting.wav \
  --lang zh --num-speakers 3 \
  --speakers "Alice,Bob,Carol" \
  --hotwords hotwords.txt \
  --speaker-context speaker-context.json
```

## 音频预处理

FunASR 最适合 16 kHz 单声道音频。建议优先使用 FLAC：它是无损格式，文件大小约为 WAV 的一半，FunASR 可通过 soundfile 原生读取。

```bash
# 推荐：无损且体积较小的 FLAC
ffmpeg -i recording.m4a -ar 16000 -ac 1 -sample_fmt s16 meeting.flac

# 备选：无损但较大的 WAV
ffmpeg -i recording.m4a -ar 16000 -ac 1 meeting.wav
```

转换 FLAC 时应使用 `-sample_fmt s16`。省略后，ffmpeg 可能输出 24 位样本（s32/24bit），文件大小会翻倍，却不会提升 ASR 效果。

### 4 小时 14 分会议的格式对比

| 格式 | 大小 | 质量 | FunASR 支持 |
|---|---:|---|---|
| **FLAC，16 kHz 单声道 s16** | **219 MB** | 无损 | soundfile 原生读取 |
| WAV，16 kHz 单声道 | 465 MB | 无损 | soundfile 原生读取 |
| Opus，32 kbps | 54 MB | 有损 | soundfile 原生读取 |
| M4A/AAC，原始 48 kHz | 173 MB | 源文件 | 通过 librosa |
| M4A/AAC，16 kHz 32 kbps | 60 MB | 有损 | 通过 librosa |

FunASR 接受常见音频格式。FLAC 在质量、大小和原生读取能力之间最均衡。

长录音不应提前切分。FunASR 可以处理任意长度文件，而分开处理会破坏不同分块之间的说话人一致性。

## 断点与恢复

处理管线支持从中断位置继续：

- 阶段 1 输出：`<文件名>_raw_transcript.json`，使用 `--skip-transcribe` 跳过 ASR；
- 阶段 3 缓存：`<文件名>_llm_cache/chunk_NNN.txt`，已清理分块会自动复用。缓存默认保留，完成后可用 `--clean-cache` 删除；
- MiMo 逐段断点：`<文件名>_mimo_partial.json`，使用完全相同的配置配合 `--resume-mimo` 恢复。

## 模型缓存

FunASR 中文预设约 3 GB，首次运行时从 ModelScope 下载并缓存到 `~/.cache/modelscope/hub/`。EC2 和临时云主机被替换后会丢失缓存，需要再次下载约两分钟。

将缓存放到持久存储：

```bash
# 推荐：命令行参数
audio-transcriber meeting.flac --model-cache-dir /data/modelscope-cache ...

# 环境变量
MODELSCOPE_CACHE=/data/modelscope-cache audio-transcriber meeting.flac ...
```

## 说话人上下文 JSON

`--speaker-context` 帮助 LLM 识别说话人并修复 ASR 错误：

```json
{
  "Alice": "Discussed Q1 revenue targets, mentioned Chicago office relocation",
  "Bob": "Presented the new CI/CD pipeline, uses Terraform and ArgoCD",
  "Carol": "HR updates, mentioned hiring freeze and new PTO policy"
}
```

每个 LLM 清理分块都会在系统提示词中收到这份上下文。

## 仅 CPU 或低内存机器

资源受限的机器在处理 2 小时以上录音时常见两类问题：执行环境超时，以及内存不足。二者都可能导致进程中途被杀死，来不及写出结果。

### 问题一：命令执行超时

交互式 Shell、CI 和远程执行环境常设置 2–10 分钟超时，而 4 小时录音使用 CPU 可能需要 1.5–2 小时。

建议把耗时的 ASR 阶段从当前会话中分离，并使用 `--skip-llm`。阶段 3 主要受网络限制，可在 ASR 完成后使用 `--skip-transcribe` 继续。

在 systemd 主机上优先使用 `systemd-run`：

```bash
systemd-run --user --unit=transcribe-job \
  --working-directory=/tmp \
  -E MODELSCOPE_CACHE=/data/modelscope-cache \
  bash -c 'source /path/to/.venv/bin/activate && \
    audio-transcriber /tmp/meeting.flac \
    --lang zh --num-speakers 9 --skip-llm > /tmp/transcribe.log 2>&1'

systemctl --user status transcribe-job.service
tail -f /tmp/transcribe.log
ls -lh /tmp/*-transcript.md /tmp/*_raw_transcript.json
```

`systemd-run` 会创建独立于当前会话的临时服务，可以跨会话重置和命令超时继续运行。

> **警告：** `systemd-run` 使用独立挂载命名空间，父会话中的 FUSE 挂载（rclone、sshfs、Google Drive 等）通常不可见。运行前应把音频、热词、说话人上下文和参考文档复制到 `/tmp` 等本地文件系统，并设置真实的工作目录。

> **注意：** `systemd-run --user` 需要用户级 systemd 实例；未启用 `loginctl enable-linger` 的容器或云主机中可能不可用。

通用备选方案是 `nohup`：

```bash
nohup bash -c 'source .venv/bin/activate && audio-transcriber meeting.flac \
  --lang zh --num-speakers 9 --skip-llm' > transcribe.log 2>&1 &

echo $!
tail -f transcribe.log
```

### 问题二：不超过 8 GB 内存时被 OOM Killer 终止

`zh` 预设同时加载 SeACo-Paraformer、VAD、标点和 CAM++。处理 4 小时录音时，峰值常驻内存可能超过 7 GB。无交换空间的机器可能被 OOM Killer 直接终止。

可先增加交换空间：

```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 完成后可删除
sudo swapoff /swapfile && sudo rm /swapfile
```

也可使用 `zh-basic`。它比 `zh` 少加载一个 SeACo 相关组件，峰值内存约降低 1–1.5 GB，代价是没有热词偏置：

```bash
audio-transcriber meeting.flac --lang zh-basic --num-speakers 9 --skip-llm
```

8 GB 内存加 4 GB 交换空间并使用 `zh-basic`，通常可以可靠处理 4 小时以上录音。

推荐流程：

```bash
# 1. 内存不超过 8 GB 时增加交换空间
sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile \
  && sudo mkswap /swapfile && sudo swapon /swapfile

# 2. 脱离当前会话执行转录
nohup bash -c 'source .venv/bin/activate && audio-transcriber meeting.flac \
  --lang zh-basic --num-speakers 9 --skip-llm' > transcribe.log 2>&1 &

# 3. 查看进度
tail -f transcribe.log

# 4. ASR 完成后继续 LLM 清理
audio-transcriber meeting.flac --skip-transcribe
```

## 播客转录

播客和访谈使用相同引擎，但工作方式与多人会议略有不同：

| 项目 | 会议 | 播客或访谈 |
|---|---|---|
| 说话人数 | 3–15 人以上，可能未知 | 通常 2–3 人，身份已知 |
| 语言 | 通常单一语言 | 可能由双语主持人混合使用 |
| 热词 | 参会者姓名和术语 | 节目名、嘉宾名和主题词 |
| 说话人上下文 | 基于职位和议题 | 主持人提问，嘉宾回答 |
| 说话人分离 | 难度较高 | 人数少且声线差异明显，通常更容易 |

推荐配置：

```bash
# 英文播客，两名说话人
audio-transcriber episode.flac --lang en --num-speakers 2 \
  --speakers "Host,Guest"

# 双语播客，自动检测语言切换
audio-transcriber episode.flac --lang auto --num-speakers 2 \
  --speakers "Alice,Bob"

# 中文播客，使用主题热词
audio-transcriber episode.flac --lang zh --num-speakers 3 \
  --speakers "主持人,嘉宾A,嘉宾B" \
  --hotwords "播客名 嘉宾全名 讨论主题关键词"

# 西班牙语加英语等其他混合语言
audio-transcriber episode.flac --lang whisper --num-speakers 2 \
  --speakers "Host,Guest"
```

播客使用建议：

- 始终提供 `--num-speakers`，已知的 2–3 人数量可以明显改善聚类；
- 始终提供 `--speakers`，主持人与嘉宾姓名通常事先已知；
- `--lang auto` 适合中英日韩粤语混合的纯文本输出，但 SenseVoiceSmall 不输出时间戳，因此不支持说话人分离。需要中文说话人识别时使用 `--lang zh`；
- 其他语言或大量语码转换使用 `--lang whisper`，该预设同样缺少用于说话人分离的时间戳；
- 中文播客的热词应包含节目名和嘉宾全名，英文播客通常不需要；
- 使用 `--speaker-context` 描述主持人与嘉宾的互动：

  ```json
  {
    "Alice": "Host, asks questions, introduces topics, wraps up segments",
    "Bob": "Guest, expert on topic X, shares personal anecdotes"
  }
  ```

- 播客通常使用录音棚设备，信噪比高于会议，因此说话人分离效果往往更好；
- 手机 App 下载可能只是试听或被截断，尽量从网页下载完整文件。阶段 0 能检测转换过程中的截断，但无法判断源文件本身是否已不完整。

## `--lang mimo`：小米 MiMo-V2.5-ASR 本地或 HTTP API

MiMo 是小米推出的 8B 参数、基于 LLM 的 ASR 模型，只直接输出纯文本，不提供逐句时间戳或说话人标签。`audio-transcriber` CLI 在本地推理或小米 HTTP API 外部套用相同的 VAD 与说话人聚类管线，使其输出格式与 FunASR 预设一致：

```text
阶段 1a  FSMN VAD           → [(start_ms, end_ms), ...]
阶段 1b  MiMo 识别          → 本地 asr_sft() 或 /chat/completions
阶段 1c  CAM++ + KMeans     → 每个 VAD 分段的说话人编号
```

相关文件：

- `audio_transcriber/mimo_asr.py`：编排、重试和断点；
- `audio_transcriber/mimo_api.py`：HTTP 客户端；
- `scripts/setup_mimo.sh`：仅本地模式需要的安装脚本。

`--mimo-backend local` 是兼容默认值，需要约 34 GB 本地安装数据和至少 20 GB 显存的 CUDA GPU。`--mimo-backend api` 不执行 CUDA、本地权重、MiMo 仓库和模型加载检查；VAD 与 CAM++ 仍根据 `--device` 在本地运行，并完整支持 CPU。

API 模式每次把一个 WAV 分段编码为 Base64 Data URL，使用 `api-key` 请求头和 `input_audio` 消息格式发送到 `{base_url}/chat/completions`。`<chinese>`、`<english>`、`<auto>` 分别映射为 `zh`、`en` 和 `auto`。

调用始终串行。API 客户端本身只发送一次请求，编排层仅对 HTTP 408、429、500、502、503、504、网络失败和超时进行有限重试，退避时间为 1、2、5、10 秒。HTTP 400、401、403、文件不存在、无效 JSON 和永久响应结构错误会立即失败。

API 配置优先级：

| 配置项 | 命令行 | 环境变量 | 默认值 |
|---|---|---|---|
| Base URL | `--mimo-api-base-url` | `MIMO_BASE_URL` | `https://api.xiaomimimo.com/v1` |
| 模型 | `--mimo-api-model` | `MIMO_API_MODEL` | `mimo-v2.5-asr` |
| Key | `--mimo-api-key-env NAME` | `NAME` 对应的值 | `MIMO_API_KEY` |
| 超时 | `--mimo-api-timeout` | — | 120 秒 |

API Key 只保存在内存中，并会从错误信息中脱敏，不会写入断点文件、原始转录、Markdown 或日志。

### 预计实时率

在单张 A100 40 GB 上，本地 MiMo 处理 4 小时音频的阶段 1 约需 24 分钟，RTF 约为 0.1。同一 GPU 上 `--lang zh` 的 RTF 约为 0.02–0.05。MiMo 以更慢速度换取在方言、语码转换和歌词等场景中的准确率提升。

### 使用 `--resume-mimo` 恢复

本地分段失败时保留三次尝试，并在重试间执行 `gc.collect()` 和 `torch.cuda.empty_cache()`；API 使用前述有限重试规则。

全部尝试失败后，`*_mimo_partial.json` 会记录后端、模型、Base URL、VAD 分段、已完成文本以及失败分段和时间范围。`--resume-mimo` 会在继续前校验音频 SHA-256、音频标签、后端、模型和 Base URL。旧版没有 `backend` 字段的本地断点按 `local` 处理。
