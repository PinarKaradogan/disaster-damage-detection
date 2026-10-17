import json

import pytest

from src.training.compare import POLICY_KEYS, compare_runs
from src.training.artifacts import write_json


def make_run(tmp_path, name, *, model="simple_cnn", loss="weighted_ce", sampler="none", score=.4):
    directory = tmp_path / name
    directory.mkdir()
    config = {key: "same" for key in POLICY_KEYS}
    config.update(model_name=model, loss=loss, sampler=sampler, batch_size=32, seed=42,
                  evaluation_split="val", smoke=False, parameter_count=100, pretrained=True)
    metrics = dict(best_val_macro_f1=score, class_f1=[score] * 4, best_epoch=2,
                   training_time_seconds=10., evaluation_split="val", smoke=False)
    write_json(directory / "config.json", config)
    write_json(directory / "metrics.json", metrics)
    write_json(directory / "status.json", {"status": "completed"})
    return directory


def test_imbalance_selection_and_summary(tmp_path):
    runs = [make_run(tmp_path, "weighted"),
            make_run(tmp_path, "focal", loss="focal", score=.5),
            make_run(tmp_path, "sampler", loss="ce", sampler="weighted", score=.3)]
    output = tmp_path / "validation.csv"
    result = compare_runs(runs, kind="imbalance", output_path=output)
    assert result["loss"] == "focal" and result["evaluation_split"] == "val"
    assert "val_macro_f1" in output.read_text() and "destroyed_f1" in output.read_text()
    assert json.loads(output.with_suffix(".json").read_text())["test_used"] is False


@pytest.mark.parametrize("change", ["smoke", "test", "seed", "loss"])
def test_reject_invalid_comparison(tmp_path, change):
    cnn = make_run(tmp_path, "cnn")
    resnet = make_run(tmp_path, "resnet", model="resnet34_early_fusion")
    config = json.loads((resnet / "config.json").read_text())
    config[{"smoke": "smoke", "test": "evaluation_split", "seed": "seed", "loss": "loss"}[change]] = {
        "smoke": True, "test": "test", "seed": 7, "loss": "focal",
    }[change]
    write_json(resnet / "config.json", config)
    with pytest.raises(ValueError):
        compare_runs([cnn, resnet], kind="models", output_path=tmp_path / "bad.csv")


def test_model_comparison_documents_batch_sizes(tmp_path):
    cnn = make_run(tmp_path, "cnn")
    resnet = make_run(tmp_path, "resnet", model="resnet34_early_fusion", score=.6)
    config = json.loads((resnet / "config.json").read_text())
    config["batch_size"] = 16
    write_json(resnet / "config.json", config)
    result = compare_runs([cnn, resnet], kind="models", output_path=tmp_path / "val.csv")
    assert result["batch_sizes"] == {"simple_cnn": 32, "resnet34_early_fusion": 16}
