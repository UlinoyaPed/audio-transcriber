"""Clustering patch success and failure exit semantics."""

from pathlib import Path

from audio_transcriber import patch_clustering


def _upstream_source() -> str:
    return (
        "def get_spec_embs(self, L, k_oracle):\n"
        f"        {patch_clustering.ORIGINAL_EIGSH}\n\n"
        "def p_pruning(A, n_elems):\n"
        f"{patch_clustering.ORIGINAL_PRUNING}\n"
    )


def test_patch_file_applies_and_is_idempotently_verified(tmp_path):
    target = tmp_path / "cluster_backend.py"
    target.write_text(_upstream_source(), encoding="utf-8")
    assert patch_clustering.patch_file(target) is True
    assert patch_clustering.patch_file(target) is True
    content = target.read_text(encoding="utf-8")
    assert "eigsh" in content
    assert "Vectorized" in content


def test_patch_file_fails_when_upstream_targets_changed(tmp_path):
    target = tmp_path / "cluster_backend.py"
    original = "def entirely_new_upstream_implementation():\n    pass\n"
    target.write_text(original, encoding="utf-8")
    assert patch_clustering.patch_file(target) is False
    assert target.read_text(encoding="utf-8") == original


def test_main_returns_nonzero_when_patch_validation_fails(
    tmp_path, monkeypatch
):
    target = tmp_path / "cluster_backend.py"
    target.write_text("unexpected upstream", encoding="utf-8")
    monkeypatch.setattr(patch_clustering, "find_cluster_backend", lambda: target)
    monkeypatch.setattr(
        "sys.argv", ["audio-transcriber-patch-clustering", "--yes"]
    )
    assert patch_clustering.main() == 1
