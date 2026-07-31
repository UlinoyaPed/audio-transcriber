from pathlib import Path

from audio_transcriber.model_revisions import (
    HF_MODEL_REVISIONS,
    MODELSCOPE_MODEL_REVISIONS,
)
from audio_transcriber.transcribe import MODEL_PRESETS


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_constraints_pin_every_direct_dependency():
    lines = [
        line.strip()
        for line in (ROOT / "constraints/runtime-py312.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert lines
    assert all(line.count("==") == 1 for line in lines)


def test_all_bundled_modelscope_pipeline_models_have_commit_pins():
    model_ids = set()
    for name, preset in MODEL_PRESETS.items():
        if name != "mimo":
            model_ids.add(preset["asr"])
        model_ids.add(preset["vad"])
        model_ids.add(preset["spk"])
        if preset.get("punc"):
            model_ids.add(preset["punc"])
    assert model_ids <= MODELSCOPE_MODEL_REVISIONS.keys()
    assert all(len(revision) == 40 for revision in MODELSCOPE_MODEL_REVISIONS.values())


def test_mimo_hugging_face_repositories_have_commit_pins():
    assert set(HF_MODEL_REVISIONS) == {
        "XiaomiMiMo/MiMo-V2.5-ASR",
        "XiaomiMiMo/MiMo-Audio-Tokenizer",
    }
    assert all(len(revision) == 40 for revision in HF_MODEL_REVISIONS.values())


def test_setup_uses_constraints_and_verified_flash_wheel():
    setup_env = (ROOT / "scripts/setup_env.sh").read_text()
    setup_mimo = (ROOT / "scripts/setup_mimo.sh").read_text()
    assert 'torch==$PYTORCH_VERSION' in setup_env
    assert '-c "$CONSTRAINTS_FILE"' in setup_env
    assert '-c "$CONSTRAINTS_FILE"' in setup_mimo
    assert "VERIFIED_WHEEL_SHA256=" in setup_mimo
    assert 'actual_sha=$(sha256sum "$WHEEL_TMP"' in setup_mimo
    assert '--revision "$MIMO_ASR_REVISION"' in setup_mimo
    assert '--revision "$MIMO_TOKENIZER_REVISION"' in setup_mimo
