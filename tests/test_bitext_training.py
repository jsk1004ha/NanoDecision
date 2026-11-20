import csv
import gzip
import hashlib
import json

import pytest

from nanodecision2.bitext_training import build, fingerprint


def fixture(tmp_path):
    sources = tmp_path / "sources"
    sources.mkdir()
    specs = []
    records = {
        "shop": [
            ("buy this", "place_order", "orders"),
            ("buy   this", "place_order", "orders"),
            ("built in sentence", "place_order", "orders"),
            ("cancel it", "cancel_order", "orders"),
            ("same words", "first", "x"),
            ("same words", "second", "x"),
        ],
        "travel": [
            ("cancel it", "cancel_booking", "booking"),
            ("find a flight", "search_flight", "booking"),
            ("held out sentence", "search_flight", "booking"),
        ],
    }
    for domain, rows in records.items():
        path = sources / f"{domain}.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["instruction", "intent", "category", "response"])
            for text, intent, category in rows:
                writer.writerow([text, intent, category, "excluded response"])
        data = path.read_bytes()
        specs.append({"domain": domain, "repo": f"publisher/{domain}-training",
            "revision": "a" * 40, "file": path.name, "url": "https://invalid.example/" + path.name,
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    license_path = tmp_path / "license.txt"
    license_path.write_text("license", encoding="utf-8")
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({
        "purpose": "publisher-designated training repositories only",
        "license": {"id": "CDLA-Sharing-1.0", "url": "https://cdla.dev/sharing-1-0/",
                    "sha256": hashlib.sha256(license_path.read_bytes()).hexdigest()},
        "excluded_training_fingerprints": {
            "scope": "fixture intersection only",
            "sha256": [fingerprint("built in sentence")],
        },
        "sources": specs,
    }), encoding="utf-8")
    heldout = tmp_path / "heldout.json"
    heldout.write_text(json.dumps([fingerprint("held out sentence")]), encoding="utf-8")
    return sources, recipe, heldout, license_path


def read_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def test_build_is_train_only_grounded_atomic_and_reproducible(tmp_path):
    sources, recipe, heldout, license_path = fixture(tmp_path)
    first = tmp_path / "first"
    report = build(recipe, first, source_dir=sources, heldout_path=heldout,
                   max_per_intent=0, license_path=license_path)
    rows = read_rows(first / "train.jsonl.gz")
    assert report["cleaning"] == {
        "source_rows": 9, "blank_rows": 0, "duplicates": 1,
        "built_in_excluded_rows": 1, "optional_excluded_rows": 1,
        "conflicting_groups": 1, "conflicting_rows": 2, "clean_rows": 4,
    }
    assert report["evaluation_data_fetched"] is False
    assert report["catalog"]["available_labels"] == 4
    assert report["training_exclusions"]["built_in"]["rows_removed"] == 1
    assert report["training_exclusions"]["optional"]["rows_removed"] == 1
    assert len(rows) == 4 and all(row["partition"] == "train" for row in rows)
    assert all(set(row["state"]) == {"context", "utterance"} for row in rows)
    assert all(row["state"]["context"]["service_domain"] in {"shop", "travel"} for row in rows)
    assert len({row["group_id"] for row in rows}) == 4
    assert {row["state"]["utterance"] for row in rows}.isdisjoint({"same words", "held out sentence"})
    assert all("response" not in json.dumps(row) for row in rows)
    assert (first / "LICENSE-DATA.txt").read_text() == "license"
    assert json.loads((first / "catalog.json").read_text(encoding="utf-8")) == rows[0]["criteria"]

    second = tmp_path / "second"
    again = build(recipe, second, source_dir=sources, heldout_path=heldout,
                  max_per_intent=0, license_path=license_path)
    assert report["output"]["sha256"] == again["output"]["sha256"]
    assert (first / "train.jsonl.gz").read_bytes() == (second / "train.jsonl.gz").read_bytes()


def test_candidate_subset_keeps_stable_groups_and_visible_domain(tmp_path):
    sources, recipe, _, license_path = fixture(tmp_path)
    all_out, subset_out = tmp_path / "all", tmp_path / "subset"
    build(recipe, all_out, source_dir=sources, max_per_intent=0, license_path=license_path)
    build(recipe, subset_out, source_dir=sources, candidate_count=2,
          max_per_intent=0, license_path=license_path)
    all_rows = {row["target"]: row for row in read_rows(all_out / "train.jsonl.gz")}
    subset = read_rows(subset_out / "train.jsonl.gz")
    assert len(subset) == 2
    assert all(len(row["criteria"]) == 2 and row["target"] in row["criteria"] for row in subset)
    assert all(all_rows[row["target"]]["group_id"] == row["group_id"] for row in subset)
    assert all(all_rows[row["target"]]["id"] != row["id"] for row in subset)
    assert {row["state"]["context"]["service_domain"] for row in subset} == {"shop", "travel"}


def test_per_intent_limit_is_deterministic(tmp_path):
    sources, recipe, _, license_path = fixture(tmp_path)
    output = tmp_path / "limited"
    report = build(recipe, output, source_dir=sources, max_per_intent=1, license_path=license_path)
    rows = read_rows(output / "train.jsonl.gz")
    assert report["output"]["rows"] == 4
    assert len({row["target"] for row in rows}) == 4


def test_hash_failure_leaves_no_partial_output(tmp_path):
    sources, recipe, _, license_path = fixture(tmp_path)
    document = json.loads(recipe.read_text())
    document["sources"][0]["sha256"] = "0" * 64
    recipe.write_text(json.dumps(document), encoding="utf-8")
    output = tmp_path / "result"
    with pytest.raises(ValueError, match="Pinned source mismatch"):
        build(recipe, output, source_dir=sources, license_path=license_path)
    assert not output.exists()
    assert not list(tmp_path.glob("result.partial-*"))


def test_rejects_non_train_recipe_and_existing_output(tmp_path):
    sources, recipe, _, license_path = fixture(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        build(recipe, output, source_dir=sources, license_path=license_path)
    bad = json.loads(recipe.read_text())
    bad["purpose"] = "mixed train and test"
    recipe.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="train-only"):
        build(recipe, tmp_path / "out", source_dir=sources, license_path=license_path)
