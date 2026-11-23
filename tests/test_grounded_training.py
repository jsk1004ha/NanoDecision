import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from nanodecision2.grounded_training import (
    WIDTHS, catalog, contrast_groups, generate, near_neighbors, ranked_group_candidates, rows,
)
from nanodecision2.schema import gold_target, normalize_question
from nanodecision2.config import ModelConfig
from nanodecision2.data import prepare_cache


def test_catalog_and_nested_candidate_sets_are_distinct_and_hard():
    records = catalog()
    assert len(records) == len({item["en"] for item in records}) == len({item["ko"] for item in records}) == 1000
    by_id = {item["id"]: item for item in records}
    for group_id, clients, requests in contrast_groups():
        golds = [by_id[f"{client}__{asset}__{action}"] for client in clients for asset, action in requests]
        previous = set()
        for width in (32, 100):
            selected = set(ranked_group_candidates(group_id, golds, records, width, 17))
            assert len(selected) == width and {item["id"] for item in golds} <= selected and previous <= selected
            for gold in golds:
                near = near_neighbors(gold, records)
                per_axis = [len(selected & set(near[i * 9:(i + 1) * 9])) for i in range(3)]
                assert min(per_axis) >= (4 if width == 32 else 9)
            previous = selected


def test_factorials_cover_catalog_and_change_target_on_each_axis():
    groups = list(contrast_groups())
    targets = set()
    for _, clients, requests in groups:
        cell = {f"{client}__{asset}__{action}" for client in clients for asset, action in requests}
        assert len(cell) == 4
        targets.update(cell)
    assert len(groups) == 250 and len(targets) == 1000
    sample = [row for row in rows(32, ("en",), 1, 9) if row["context_variant"] == 0]
    assert len(sample) == 4 and len({row["group_id"] for row in sample}) == 1
    assert len({tuple(row["criteria"]) for row in sample}) == 1
    by_request, by_client = {}, {}
    for row in sample:
        state = row["state"]
        by_request.setdefault(state["utterance"], set()).add(row["target"])
        by_client.setdefault(state["context"]["current_service_account"], set()).add(row["target"])
        assert row["target"] not in json.dumps(state, ensure_ascii=False)
    assert all(len(values) == 2 for values in (*by_request.values(), *by_client.values()))


@pytest.mark.parametrize("width", [32, 100, 200])
def test_generated_rows_are_prepare_compatible_at_supported_widths(tmp_path, width):
    out = tmp_path / f"grounded-{width}.jsonl"
    report = generate(out, width, "both", 1, 5)
    parsed = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert report["rows"] == len(parsed) == 16 and report["contrast_groups"] == 1
    for row in parsed:
        assert row["partition"] == "train" and len(row["criteria"]) == width
        assert sum(gold_target(row, normalize_question(row))) == 1


def test_same_semantics_and_group_across_widths_with_shuffled_positions():
    def key(row):
        return row["target"], row["context_variant"]
    by_width = {width: {key(row): row for row in rows(width, ("ko",), 1, 31)} for width in WIDTHS}
    assert all(value.keys() == by_width[32].keys() for value in by_width.values())
    for row_key, base in by_width[32].items():
        prior, positions = set(), []
        for width in WIDTHS:
            row = by_width[width][row_key]
            assert (row["state"], row["target"], row["group_id"]) == (base["state"], base["target"], base["group_id"])
            assert prior <= set(row["criteria"])
            prior = set(row["criteria"])
            positions.append(list(row["criteria"]).index(row["target"]))
        assert len(set(positions)) > 1


def test_irrelevant_context_and_all_factorial_cells_share_ordered_candidates():
    sample = list(rows(100, ("ko",), 1, 47))
    assert len(sample) == 8
    assert len({tuple(row["criteria"]) for row in sample}) == 1
    by_target = {}
    for row in sample:
        by_target.setdefault(row["target"], []).append(row)
    assert len(by_target) == 4
    for variants in by_target.values():
        assert len(variants) == 2
        assert variants[0]["state"]["utterance"] == variants[1]["state"]["utterance"]
        assert variants[0]["state"]["context"]["current_service_account"] == variants[1]["state"]["context"]["current_service_account"]
        assert variants[0]["state"]["context"] != variants[1]["state"]["context"]


def test_generation_is_reproducible_and_refuses_overwrite(tmp_path):
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a = generate(first, 32, "ko", 2, 11)
    b = generate(second, 32, "ko", 2, 11)
    assert a["sha256"] == b["sha256"] and first.read_bytes() == second.read_bytes()
    with pytest.raises(FileExistsError):
        generate(first, 32, "ko", 2, 11)
    orphan = tmp_path / "orphan.jsonl"
    sidecar = orphan.with_suffix(".jsonl.manifest.json")
    sidecar.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        generate(orphan, 32, "ko", 1, 11)
    assert not orphan.exists() and sidecar.read_text(encoding="utf-8") == "keep"


def test_gzip_generation_is_byte_reproducible_across_output_names(tmp_path):
    first, second = tmp_path / "a.jsonl.gz", tmp_path / "renamed.jsonl.gz"
    a = generate(first, 32, "ko", 2, 11)
    b = generate(second, 32, "ko", 2, 11)
    assert a["sha256"] == b["sha256"]
    assert first.read_bytes() == second.read_bytes()


def test_concurrent_same_path_generation_is_fail_closed(tmp_path):
    out = tmp_path / "concurrent.jsonl"
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(generate, out, 32, "en", 2, seed) for seed in (11, 19)]
    outcomes = []
    for future in futures:
        try:
            outcomes.append(("ok", future.result()))
        except FileExistsError:
            outcomes.append(("exists", None))
    assert sorted(kind for kind, _ in outcomes) == ["exists", "ok"]
    report = next(value for kind, value in outcomes if kind == "ok")
    manifest = json.loads(out.with_suffix(".jsonl.manifest.json").read_text(encoding="utf-8"))
    assert manifest == report
    assert manifest["sha256"] == report["sha256"]
    assert not list(tmp_path.glob("*.partial-*"))


def test_two_gzip_widths_pass_one_real_prepare_path(tmp_path):
    source = tmp_path / "init"
    source.mkdir()
    (source / "init.json").write_text('{"smoke_only":true}', encoding="utf-8")
    (source / "config.json").write_text(json.dumps(ModelConfig.smoke().to_dict()), encoding="utf-8")
    narrow = tmp_path / "grounded-32.jsonl.gz"
    wide = tmp_path / "grounded-1000.jsonl.gz"
    first = generate(narrow, 32, "en", 1, 23)
    second = generate(wide, 1000, "en", 1, 23)
    manifest = prepare_cache([narrow, wide], tmp_path / "cache", source, max_state=512, max_pair=256)
    assert first["rows"] + second["rows"] == manifest["counts"]["train"] == 16
    assert manifest["duplicates_removed"] == 0
    assert first["compressed"] and second["compressed"]


def test_manifest_publish_failure_rolls_back_data(tmp_path, monkeypatch):
    out = tmp_path / "rollback.jsonl"
    manifest = out.with_suffix(".jsonl.manifest.json")
    original = os.link

    def fail_manifest(source, target):
        if str(target) == str(manifest):
            raise OSError("injected manifest publication failure")
        return original(source, target)

    monkeypatch.setattr(os, "link", fail_manifest)
    with pytest.raises(OSError, match="injected"):
        generate(out, 32, "en", 1, 53)
    assert not out.exists() and not manifest.exists()
    assert not list(tmp_path.glob("*.partial-*"))
