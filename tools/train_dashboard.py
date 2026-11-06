#!/usr/bin/env python3
"""Read-only, dependency-free training monitor for NanoDecision 0.3.1.

Run separately from training.  This module deliberately lives outside
nanodecision2/ so enabling the dashboard does not change the pipeline's
source fingerprint and invalidate an existing Auto5M resume.
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
from urllib.parse import urlsplit


MAX_HISTORY_BYTES = 12 * 1024 * 1024
MAX_HISTORY_POINTS = 8_000
PHASES = ("english_reference", "mixed_training", "train")
STAGES = (
    "preflight", "initialize", "data", "english_reference", "mixed_training"
)


def read_json(path: Path) -> dict:
    """Read a report while a separate process may be creating it."""
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        return result if isinstance(result, dict) else {}
    except (OSError, ValueError, UnicodeError):
        return {}


def finite(value):
    """Avoid nonstandard JSON NaN/Infinity, which JSONL logs may contain."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def compact_entry(entry: dict) -> dict | None:
    step = entry.get("step")
    if type(step) is not int or step < 0:
        return None
    val = entry.get("validation")
    val = val if isinstance(val, dict) else {}
    langs = val.get("by_language")
    langs = langs if isinstance(langs, dict) else {}

    def language_metrics(code):
        row = langs.get(code)
        row = row if isinstance(row, dict) else {}
        return {"nll": finite(row.get("nll")), "accuracy": finite(row.get("accuracy"))}

    return {
        "step": step,
        "epoch": finite(entry.get("epoch")),
        "train_loss": finite(entry.get("train_loss")),
        "val_nll": finite(val.get("nll")),
        "val_accuracy": finite(val.get("accuracy")),
        "macro_nll": finite(val.get("macro_language_nll")),
        "languages": {"en": language_metrics("en"), "ko": language_metrics("ko")},
        "english_gate": entry.get("english_gate_passed") if isinstance(entry.get("english_gate_passed"), bool) else None,
        "all_gates": entry.get("all_requested_language_gates_passed") if isinstance(entry.get("all_requested_language_gates_passed"), bool) else None,
    }


def read_history(path: Path) -> tuple[list[dict], float | None]:
    """Read only the complete tail of an actively appended JSONL file."""
    try:
        with path.open("rb") as stream:
            stamp = path.stat().st_mtime
            stream.seek(0, 2)
            start = max(0, stream.tell() - MAX_HISTORY_BYTES)
            stream.seek(start)
            if start:
                stream.readline()  # The first line may have been cut in half.
            points = []
            for line in stream:
                if not line.endswith(b"\n"):
                    continue  # Writer may still be appending this line.
                try:
                    entry = compact_entry(json.loads(line))
                except (ValueError, UnicodeError, TypeError):
                    continue
                if entry is not None:
                    points.append(entry)
            return points[-MAX_HISTORY_POINTS:], stamp
    except OSError:
        return [], None


def expected_steps(recipe: dict, data: dict, phase: str) -> int | None:
    config = recipe.get("training")
    config = config if isinstance(config, dict) else {}
    counts = data.get("counts")
    counts = counts if isinstance(counts, dict) else {}
    languages = data.get("by_language")
    languages = languages if isinstance(languages, dict) else {}
    if phase == "english_reference":
        rows = languages.get("train:en")
        epochs = config.get("english_reference_epochs")
    elif phase == "mixed_training":
        rows = counts.get("train")
        epochs = config.get("epochs")
    else:
        return None  # Standalone CLI training has its own settings.
    batch = config.get("batch_size")
    accumulation = config.get("accumulation")
    if any(type(v) is not int or v < 1 for v in (rows, epochs, batch, accumulation)):
        return None
    batches = math.ceil(rows / batch)
    return math.ceil(batches / accumulation) * epochs


def collect_snapshot(work: Path) -> dict:
    """Use only known, read-only filenames under the chosen work directory."""
    work = Path(work)
    pipeline = read_json(work / "pipeline_state.json")
    recipe = read_json(work / "recipe.lock.json")
    data = read_json(work / "data_report.json")
    final = read_json(work / "FINAL_REPORT.json")
    active_stage = pipeline.get("current_stage")
    stages = pipeline.get("stages")
    stages = stages if isinstance(stages, dict) else {}
    phase_data = {}
    for name in PHASES:
        phase_dir = work if name == "train" else work / name
        history_path = phase_dir / "history.jsonl"
        report_path = phase_dir / "training_report.json"
        if name == "train" and not history_path.exists() and not report_path.exists():
            continue
        history, updated = read_history(history_path)
        report = read_json(report_path)
        stage_info = stages.get(name)
        stage_info = stage_info if isinstance(stage_info, dict) else {}
        complete = bool(report.get("completed")) or stage_info.get("status") == "complete"
        phase_data[name] = {
            "history": history,
            "updated": updated,
            "complete": complete,
            "interrupted": bool(report.get("interrupted")),
            "expected_steps": expected_steps(recipe, data, name),
            "reported_steps": finite(report.get("optimizer_steps_total")),
            "best_selection_loss": finite(report.get("best_selection_loss")),
            "selection_metric": report.get("selection_metric") if isinstance(report.get("selection_metric"), str) else None,
        }

    if active_stage in ("english_reference", "mixed_training"):
        selected = active_stage
    elif phase_data:
        selected = next((key for key in ("mixed_training", "english_reference", "train")
                         if phase_data.get(key, {}).get("history")), next(iter(phase_data)))
    else:
        selected = "english_reference"

    timeline = [
        {"name": name, "status": stages[name].get("status", "pending")
         if isinstance(stages.get(name), dict) else "pending"}
        for name in STAGES
    ]
    downstream = [v for k, v in stages.items() if k not in STAGES and isinstance(v, dict)]
    timeline.append({"name": "finalize", "status": (
        "complete" if final.get("status") == "complete" else
        "running" if downstream and any(v.get("status") == "running" for v in downstream) else
        "failed" if any(v.get("status") == "failed" for v in downstream) else
        "pending"
    )})
    return {
        "workdir": str(work.resolve()),
        "pipeline_status": pipeline.get("status", "unknown"),
        "active_stage": active_stage,
        "selected_phase": selected,
        "stages": timeline,
        "phases": phase_data,
        "quality_status": final.get("quality_status"),
        "smoke_only": bool(recipe.get("smoke_only", data.get("smoke_only", False))),
    }


def handler_for(work: Path):
    html = Path(__file__).with_suffix(".html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = urlsplit(self.path).path
            if route in ("/", "/index.html"):
                body, mime = html, "text/html; charset=utf-8"
            elif route == "/api/snapshot":
                body = json.dumps(collect_snapshot(work), ensure_ascii=False, allow_nan=False).encode("utf-8")
                mime = "application/json; charset=utf-8"
            else:
                self.send_error(404, "Not found")
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            if urlsplit(self.path).path != "/api/snapshot":
                super().log_message(fmt, *args)

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description="NanoDecision read-only live training dashboard (stdlib only)")
    parser.add_argument("--workdir", required=True, type=Path, help="Auto5M workdir or a standalone nd2 train --out folder")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: local machine only)")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if not args.workdir.is_dir():
        parser.error(f"Not a directory: {args.workdir}")
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    try:
        with ThreadingHTTPServer((args.host, args.port), handler_for(args.workdir)) as server:
            print(f"Training dashboard: http://{args.host}:{server.server_port}/", flush=True)
            print(f"Reading: {args.workdir.resolve()} (read-only; Ctrl+C to stop)", flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    except OSError as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
