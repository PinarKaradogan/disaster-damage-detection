"""Phase 6 one-shot test-evaluation guards and outputs on synthetic data (real test data never read)."""
import json
import subprocess

import numpy as np
import pytest
import torch

from src.evaluation.evaluate_test import (
    PROTOCOL, evaluate_classifier, output_path, protocol_commit, verify_checkpoint,
)
from src.evaluation.two_stage import harmonic_mean, missed_as_false_negative
from src.training.artifacts import file_sha256
from src.training.config import TrainingConfig
from src.training.experiment import run_experiment
from test_training_data import synthetic_manifest  # noqa: F401  (pytest fixture)

SHA_A = "a" * 64


def write_protocol(root, rows):
    path = root / PROTOCOL
    path.parent.mkdir(parents=True, exist_ok=True)
    table = "\n".join(f"| role | `{run}` | `{sha}` | x |" for run, sha in rows)
    path.write_text("# protocol\n\n| Rol | Koşu | SHA | Val |\n|---|---|---|---|\n" + table + "\n"
                    "| CNN | `pending_run` | later* | x |\n", encoding="utf-8")
    return path


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True)


def test_protocol_parsing_registers_only_rows_with_a_sha(tmp_path):
    write_protocol(tmp_path, [("run_a", SHA_A)])
    _, registered = protocol_commit(tmp_path, check_git=False)
    assert registered == {"run_a": SHA_A}


def test_protocol_must_be_committed_and_clean(tmp_path):
    write_protocol(tmp_path, [("run_a", SHA_A)])
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "t@example.com")
    git(tmp_path, "config", "user.name", "t")
    with pytest.raises(RuntimeError, match="not committed"):
        protocol_commit(tmp_path)
    git(tmp_path, "add", PROTOCOL)
    git(tmp_path, "commit", "-q", "-m", "protocol")
    commit, registered = protocol_commit(tmp_path)
    assert len(commit) == 40 and registered == {"run_a": SHA_A}
    (tmp_path / PROTOCOL).write_text("# changed after commit\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="uncommitted"):
        protocol_commit(tmp_path)


def test_checkpoint_sha_must_match_protocol(tmp_path):
    checkpoint = tmp_path / "best.pth"
    checkpoint.write_bytes(b"weights")
    with pytest.raises(ValueError, match="not pre-registered"):
        verify_checkpoint("pending_run", checkpoint, {"run_a": SHA_A})
    with pytest.raises(ValueError, match="differs"):
        verify_checkpoint("run_a", checkpoint, {"run_a": SHA_A})
    assert verify_checkpoint("run_a", checkpoint, {"run_a": file_sha256(checkpoint)}) == file_sha256(checkpoint)


def test_output_is_one_shot(tmp_path):
    path = output_path("classifier", "run_a", tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    with pytest.raises(FileExistsError, match="one-shot"):
        output_path("classifier", "run_a", tmp_path)


def test_classifier_test_evaluation_end_to_end(synthetic_manifest, tmp_path):  # noqa: F811
    root, manifest = synthetic_manifest
    config = TrainingConfig("simple_cnn", batch_size=2, input_size=16, epochs=1, amp=False)
    run_experiment(config, manifest_path=manifest, experiment_name="cnn", results_root=root / "results",
                   runs_root=root / "runs", project_root=root, device="cpu", loss="focal")
    write_protocol(root, [("cnn", file_sha256(root / "results/cnn/best.pth"))])
    report = evaluate_classifier("cnn", project_root=root, num_workers=0, check_git=False, device="cpu")
    assert report["evaluation_split"] == "test" and report["test_used"] is True
    assert report["test_samples"] == 4 and len(report["test"]["class_f1"]) == 4
    assert report["best_pth_sha256"] == file_sha256(root / "results/cnn/best.pth")
    saved = json.loads((root / "results/phase6_test/classifier_cnn.json").read_text())
    assert saved["test"]["macro_f1"] == report["test"]["macro_f1"]
    rows = (root / "results/phase6_test/classifier_cnn.csv").read_text().splitlines()
    assert rows[0] == "image_id,uid,disaster,label,pred,p0,p1,p2,p3" and len(rows) == 5
    assert [row.split(",")[0] for row in rows[1:]] == [f"test_{i}" for i in range(4)]
    with pytest.raises(FileExistsError):
        evaluate_classifier("cnn", project_root=root, num_workers=0, check_git=False, device="cpu")
    torch.save({"tampered": True}, root / "results/cnn/best.pth")
    (root / "results/phase6_test/classifier_cnn.json").unlink()
    (root / "results/phase6_test/classifier_cnn.csv").unlink()
    with pytest.raises(ValueError, match="differs from the protocol"):
        evaluate_classifier("cnn", project_root=root, num_workers=0, check_git=False, device="cpu")


def test_xview2_helpers():
    assert harmonic_mean([0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)
    assert harmonic_mean([1.0, 0.5, 1.0, 0.5]) == pytest.approx(4 / 6)
    assert harmonic_mean([0.9, 0.0, 0.9, 0.9]) == 0.0
    labels = np.array([0, 0, 1, 1])
    preds = np.array([0, 0, 1, 0])
    detected = np.array([True, True, True, False])      # last building (class 1) missed
    result = missed_as_false_negative(labels, preds, detected)
    # class 0: TP 2, FP 0 (the missed building is NOT a false positive for class 0) -> F1 1.0
    # class 1: TP 1, FN 1 -> F1 2/3
    assert result["class_f1"][0] == pytest.approx(1.0)
    assert result["class_f1"][1] == pytest.approx(2 / 3)
