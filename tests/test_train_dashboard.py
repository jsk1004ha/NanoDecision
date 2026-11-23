"""Tests for the dependency-free, read-only training dashboard."""

from http.server import ThreadingHTTPServer
import json
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.request import urlopen

from tools.train_dashboard import collect_snapshot, expected_steps, handler_for, read_history


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_history_reads_only_complete_lines_and_ignores_nonfinite(tmp_path):
    history = tmp_path / "history.jsonl"
    history.write_bytes(
        b'{"step":1,"epoch":1,"train_loss":1.2}\n'
        b'{not json}\n'
        b'{"step":2,"epoch":1,"train_loss":0.9,"validation":{"nll":Infinity,'
        b'"accuracy":0.5,"by_language":{"en":{"nll":0.7}}}}\n'
        b'{"step":3,"train_loss":0.8'  # Other process has not finished writing.
    )
    points, timestamp = read_history(history)
    assert [p["step"] for p in points] == [1, 2]
    assert points[1]["val_nll"] is None
    assert points[1]["languages"]["en"]["nll"] == 0.7
    assert points[1]["val_accuracy"] == 0.5
    assert timestamp is not None


def test_planned_steps_match_training_accumulation():
    recipe = {"training": {"batch_size": 3, "accumulation": 2, "english_reference_epochs": 1, "epochs": 2}}
    data = {"counts": {"train": 24}, "by_language": {"train:en": 11}}
    assert expected_steps(recipe, data, "english_reference") == 2
    assert expected_steps(recipe, data, "mixed_training") == 8
    assert expected_steps(recipe, data, "train") is None
    assert expected_steps({}, {}, "english_reference") is None


def test_pipeline_snapshot_has_two_independent_training_phases(tmp_path):
    work = tmp_path / "run"
    work.mkdir()
    save_json(work / "recipe.lock.json", {"smoke_only": True, "training": {
        "batch_size": 2, "accumulation": 2, "epochs": 2, "english_reference_epochs": 1}})
    save_json(work / "data_report.json", {"counts": {"train": 120}, "by_language": {"train:en": 60}})
    save_json(work / "pipeline_state.json", {"status": "running", "current_stage": "mixed_training",
        "stages": {"preflight": {"status": "complete"}, "english_reference": {"status": "complete"},
                   "mixed_training": {"status": "running"}}})
    ref = work / "english_reference"
    mixed = work / "mixed_training"
    ref.mkdir()
    mixed.mkdir()
    (ref / "history.jsonl").write_text('{"step":15,"train_loss":0.6}\n')
    (mixed / "history.jsonl").write_text('{"step":5,"train_loss":0.3,"validation":'
        '{"nll":0.45,"by_language":{"en":{"nll":0.5},"ko":{"nll":0.4}}}}\n')
    snapshot = collect_snapshot(work)
    assert snapshot["selected_phase"] == "mixed_training"
    assert snapshot["smoke_only"]
    assert snapshot["phases"]["english_reference"]["expected_steps"] == 15
    assert snapshot["phases"]["mixed_training"]["expected_steps"] == 60
    assert snapshot["phases"]["mixed_training"]["history"][0]["languages"]["ko"]["nll"] == 0.4
    assert snapshot["stages"][0]["status"] == "complete"
    assert snapshot["stages"][-1]["name"] == "finalize"


def test_standalone_training_out_directory(tmp_path):
    (tmp_path / "history.jsonl").write_text('{"step":1,"train_loss":0.8}\n')
    save_json(tmp_path / "training_report.json", {"completed": True, "optimizer_steps_total": 1,
        "best_selection_loss": 0.5, "selection_metric": "nll"})
    snapshot = collect_snapshot(tmp_path)
    assert snapshot["selected_phase"] == "train"
    assert snapshot["phases"]["train"]["complete"]
    assert snapshot["phases"]["train"]["expected_steps"] is None
    assert snapshot["phases"]["train"]["best_selection_loss"] == 0.5


def test_http_serves_dashboard_and_read_only_json(tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(tmp_path))
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urlopen(base + "/") as response:
            html = response.read().decode("utf-8")
            assert response.headers["Cache-Control"] == "no-store"
            assert "실제 학습 루프" in html
            assert "language-chart" in html
        with urlopen(base + "/api/snapshot") as response:
            payload = json.loads(response.read())
            assert payload["phases"]["english_reference"]["history"] == []
            assert payload["phases"]["mixed_training"]["history"] == []
            assert payload["pipeline_status"] == "unknown"
        try:
            urlopen(base + "/../../secret")
            assert False, "Unknown paths must not be served"
        except HTTPError as exc:
            assert exc.code == 404
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
