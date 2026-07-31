"""Immutable upstream model revisions validated by this project."""

from typing import Optional

MODELSCOPE_MODEL_REVISIONS = {
    "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch":
        "0141367fdc9b6ba58b0442ef34bceb56a6c1789c",
    "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch":
        "37851ca81e5df7d2532df68ae87cafc11b70aa7e",
    "iic/speech_paraformer-large-vad-punc_asr_nat-en-16k-common-vocab10020":
        "97ce0bc9b0178aac03008c3afe8271cdb534228f",
    "iic/SenseVoiceSmall":
        "7bf452403abd7353a300cd760f7adae7701c92c1",
    "iic/Whisper-large-v3-turbo":
        "1d04d0d7447f5a81f1a9addc727bf64f12ab8db0",
    "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch":
        "f9a8b8274674755d925277e27063869038d41515",
    "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch":
        "d488ddd5985989501d57bce7f7a4d4dcfca779d0",
    "iic/speech_campplus_sv_zh-cn_16k-common":
        "a045b2afcaa9c3049c98a9215a2bc274407ab237",
}

HF_MODEL_REVISIONS = {
    "XiaomiMiMo/MiMo-V2.5-ASR":
        "98641d537df521ac6df05f74090475694d9510b7",
    "XiaomiMiMo/MiMo-Audio-Tokenizer":
        "5df9914f72d3acda1320d7fecde7d91622edb0c1",
}


def modelscope_revision(model_id: str) -> Optional[str]:
    """Return a bundled model's revision, or ``None`` for a custom model."""
    return MODELSCOPE_MODEL_REVISIONS.get(model_id)


def hf_revision(repo_id: str) -> str:
    """Return the immutable revision for a bundled Hugging Face model."""
    try:
        return HF_MODEL_REVISIONS[repo_id]
    except KeyError as exc:
        raise ValueError(
            f"No pinned Hugging Face revision for bundled model {repo_id!r}"
        ) from exc
