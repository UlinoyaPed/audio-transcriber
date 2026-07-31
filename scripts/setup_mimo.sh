#!/usr/bin/env bash
# Install MiMo-V2.5-ASR: clone the reference repo, install flash-attn, and
# download HF weights. Called automatically by setup_env.sh when
# INSTALL_MIMO=1, or can be run standalone after scripts/setup_env.sh.
#
# Environment:
#   VENV_DIR             — path to the active venv (default: .venv)
#   MIMO_WEIGHTS_PATH    — cache dir for MiMo weights (default: $HF_HOME
#                          or ~/.cache/huggingface)

set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
MIMO_WEIGHTS_PATH="${MIMO_WEIGHTS_PATH:-${HF_HOME:-$HOME/.cache/huggingface}}"
MIMO_REPO_URL="https://github.com/XiaomiMiMo/MiMo-V2.5-ASR.git"
MIMO_REPO_DIR="$VENV_DIR/mimo"
# Pinned to a known-good commit validated in e2e against our MiMoAudio
# kwarg contract. Upgrading this pin is a deliberate act: verify
# MimoAudio.__init__ signature (expects mimo_audio_tokenizer_path=, not
# tokenizer_path=) and that the declared deps still satisfy einops/addict
# before bumping. Users can override to trial a newer upstream.
MIMO_PINNED_COMMIT="${MIMO_PINNED_COMMIT:-210ef16815b187e05ccc38d627af0d61677afe88}"
MIMO_ASR_REVISION="98641d537df521ac6df05f74090475694d9510b7"
MIMO_TOKENIZER_REVISION="5df9914f72d3acda1320d7fecde7d91622edb0c1"
CONSTRAINTS_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/constraints/runtime-py312.txt"

echo "=== MiMo-V2.5-ASR Install ==="
echo "  venv:     $VENV_DIR"
echo "  weights:  $MIMO_WEIGHTS_PATH"
echo ""

# Require active venv
if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: venv not found at $VENV_DIR. Run scripts/setup_env.sh first."
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# 1. Clone MiMo repo pinned to a known-good commit (supply-chain hygiene).
#    Idempotent: if the repo is already at the pinned SHA we skip; if it's
#    checked out to a different SHA we fast-forward to the pin.
if [ ! -d "$MIMO_REPO_DIR/.git" ]; then
    echo "[1/4] Cloning $MIMO_REPO_URL at $MIMO_PINNED_COMMIT into $MIMO_REPO_DIR..."
    git clone --filter=blob:none --no-checkout "$MIMO_REPO_URL" "$MIMO_REPO_DIR"
    git -C "$MIMO_REPO_DIR" fetch --depth 1 origin "$MIMO_PINNED_COMMIT"
    git -C "$MIMO_REPO_DIR" checkout --detach "$MIMO_PINNED_COMMIT"
else
    current=$(git -C "$MIMO_REPO_DIR" rev-parse HEAD 2>/dev/null || echo "")
    if [ "$current" = "$MIMO_PINNED_COMMIT" ]; then
        echo "[1/4] MiMo repo already at pinned commit ${MIMO_PINNED_COMMIT:0:10} — skipping."
    else
        echo "[1/4] MiMo repo at ${current:0:10}, fast-forwarding to ${MIMO_PINNED_COMMIT:0:10}..."
        git -C "$MIMO_REPO_DIR" fetch --depth 1 origin "$MIMO_PINNED_COMMIT"
        git -C "$MIMO_REPO_DIR" checkout --detach "$MIMO_PINNED_COMMIT"
    fi
fi

# Post-checkout verification: HEAD must equal the pinned SHA. Guards
# against tampered on-disk state or a --filter race. Abort loudly rather
# than installing from an unverified tree.
resolved=$(git -C "$MIMO_REPO_DIR" rev-parse HEAD)
if [ "$resolved" != "$MIMO_PINNED_COMMIT" ]; then
    echo "ERROR: MiMo repo HEAD ($resolved) does not match pinned commit ($MIMO_PINNED_COMMIT)."
    echo "  Refusing to proceed. Remove $MIMO_REPO_DIR and re-run to reinstall."
    exit 1
fi

# 2. Install MiMo's Python dependencies. Upstream requirements.txt is
# incomplete — the runtime code imports einops (via internal audio modules)
# and addict (via the 3D-Speaker gender classifier path) without declaring
# them. Install both alongside the declared deps so first-run inference
# doesn't fail on ModuleNotFoundError.
if [ -f "$MIMO_REPO_DIR/requirements.txt" ]; then
    echo "[2/4] Installing MiMo requirements..."
    pip install -q -c "$CONSTRAINTS_FILE" -r "$MIMO_REPO_DIR/requirements.txt"
else
    echo "  WARNING: $MIMO_REPO_DIR/requirements.txt missing — skipping."
fi
echo "  Installing additional runtime deps (upstream missed these): einops, addict"
pip install -q -c "$CONSTRAINTS_FILE" einops addict

# 3. Install flash-attn. Only the exact wheel validated by the project's E2E
# run is accepted, and it must match its committed SHA-256. Other combinations
# build the pinned source release instead of executing an unverified binary.
if python3 -c "import flash_attn" 2>/dev/null; then
    echo "[3/4] flash-attn already installed — skipping."
else
    FA_VER="2.7.4.post1"
    echo "[3/4] Detecting verified flash-attn wheel for installed torch..."
    read -r TORCH_MINOR ABI <<EOF_DETECT
$(python3 -c "import torch; v=torch.__version__.split('+')[0].rsplit('.',1)[0]; print(v, 'TRUE' if torch.compiled_with_cxx11_abi() else 'FALSE')")
EOF_DETECT
    PY_MINOR=$(python3 -c "import sys; print(f'cp{sys.version_info[0]}{sys.version_info[1]}')")
    WHEEL_NAME="flash_attn-${FA_VER}+cu12torch${TORCH_MINOR}cxx11abi${ABI}-${PY_MINOR}-${PY_MINOR}-linux_x86_64.whl"
    WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FA_VER}/${WHEEL_NAME}"
    VERIFIED_WHEEL_NAME="flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
    VERIFIED_WHEEL_SHA256="7e0b07913d56782d0f3f2ee76bd39587557742628983f6ab9ec2527c99437476"
    echo "  torch=${TORCH_MINOR} abi=${ABI} py=${PY_MINOR}"
    echo "  wheel: ${WHEEL_NAME}"

    wheel_installed=""
    if [ "$WHEEL_NAME" = "$VERIFIED_WHEEL_NAME" ]; then
        WHEEL_TMP=$(mktemp --suffix=.whl)
        trap 'rm -f "$WHEEL_TMP"' EXIT
        if curl --fail --location --retry 2 --output "$WHEEL_TMP" "$WHEEL_URL"; then
            actual_sha=$(sha256sum "$WHEEL_TMP" | cut -d' ' -f1)
            if [ "$actual_sha" != "$VERIFIED_WHEEL_SHA256" ]; then
                echo "ERROR: flash-attn wheel SHA-256 mismatch."
                echo "  expected: $VERIFIED_WHEEL_SHA256"
                echo "  actual:   $actual_sha"
                exit 1
            fi
            pip install --no-deps "$WHEEL_TMP"
            wheel_installed=1
            echo "  Installed verified pre-built wheel."
        else
            echo "  Verified wheel download failed; trying pinned source build."
        fi
        rm -f "$WHEEL_TMP"
        trap - EXIT
    else
        echo "  No verified wheel for this combination; using pinned source build."
    fi

    if [ -z "$wheel_installed" ]; then
        if ! command -v nvcc &>/dev/null; then
            echo "ERROR: no verified pre-built wheel for this platform and nvcc not found."
            echo "  Install the CUDA toolkit to build the pinned source release:"
            echo "  https://developer.nvidia.com/cuda-toolkit"
            exit 1
        fi
        echo "  Building flash-attn==${FA_VER} from source (10–30 min)..."
        pip install "flash-attn==${FA_VER}" --no-build-isolation
    fi
fi

# 4. Download weights (idempotent — huggingface-cli skips cached files)
echo "[4/4] Downloading MiMo weights to $MIMO_WEIGHTS_PATH..."
mkdir -p "$MIMO_WEIGHTS_PATH"
HF_HOME="$MIMO_WEIGHTS_PATH" \
    python3 -m huggingface_hub.commands.huggingface_cli download \
        XiaomiMiMo/MiMo-V2.5-ASR \
        --revision "$MIMO_ASR_REVISION"
HF_HOME="$MIMO_WEIGHTS_PATH" \
    python3 -m huggingface_hub.commands.huggingface_cli download \
        XiaomiMiMo/MiMo-Audio-Tokenizer \
        --revision "$MIMO_TOKENIZER_REVISION"

echo ""
echo "=== MiMo install complete ==="
echo "  Use with: audio-transcriber <audio> --lang mimo \\"
echo "               --mimo-weights-path $MIMO_WEIGHTS_PATH"
