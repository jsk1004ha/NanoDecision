"""Build auditable, train-only Bitext intent data with visible service domains."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import sqlite3
import tempfile
import unicodedata
import urllib.request
from collections import defaultdict
from pathlib import Path


RECIPE = Path(__file__).with_name("recipes") / "bitext_training_sources.json"
LICENSE_TEXT = Path(__file__).with_name("recipes") / "CDLA-Sharing-1.0.txt"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def normalized_visible(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def fingerprint(text: str) -> str:
    words = re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold())
    return hashlib.sha256(" ".join(words).encode()).hexdigest()


def load_fingerprints(path: Path | None) -> tuple[set[str], str | None]:
    if path is None:
        return set(), None
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list) or any(
        not isinstance(item, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", item)
        for item in values
    ):
        raise ValueError("held-out fingerprints must be a JSON array of SHA-256 strings")
    return {item.casefold() for item in values}, sha256(path)


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "nanodecision2-bitext/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)


def source_path(source: dict, source_dir: Path | None, downloads: Path, allow_download: bool) -> Path:
    if source_dir is not None:
        path = source_dir / source["file"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing pinned source file: {path}")
    else:
        if not allow_download:
            raise ValueError("Pass --source-dir or --download")
        downloads.mkdir(parents=True, exist_ok=True)
        path = downloads / source["file"]
        download(source["url"], path)
    if path.stat().st_size != source["bytes"] or sha256(path) != source["sha256"]:
        raise ValueError(f"Pinned source mismatch: {source['domain']}")
    return path


def initialize_db(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.executescript("""
        PRAGMA journal_mode=DELETE;
        PRAGMA synchronous=FULL;
        CREATE TABLE records(
            domain TEXT NOT NULL, normalized TEXT NOT NULL, target TEXT NOT NULL,
            text TEXT NOT NULL, category TEXT NOT NULL, source_repo TEXT NOT NULL,
            source_revision TEXT NOT NULL, source_row INTEGER NOT NULL,
            PRIMARY KEY(domain, normalized, target)
        ) WITHOUT ROWID;
    """)
    return db


def ingest(db: sqlite3.Connection, sources: list[dict], source_dir: Path | None,
           downloads: Path, allow_download: bool, built_in_exclusions: set[str],
           optional_exclusions: set[str]) -> dict:
    csv.field_size_limit(16 * 1024 * 1024)
    stats = {"source_rows": 0, "blank_rows": 0, "duplicates": 0,
             "built_in_excluded_rows": 0, "optional_excluded_rows": 0}
    for source in sources:
        path = source_path(source, source_dir, downloads, allow_download)
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"instruction", "intent", "category"}
            if not reader.fieldnames or not required <= set(reader.fieldnames):
                raise ValueError(f"{source['domain']}: expected CSV columns {sorted(required)}")
            for source_row, row in enumerate(reader, start=2):
                stats["source_rows"] += 1
                text, intent, category = (row[name].strip() for name in ("instruction", "intent", "category"))
                if not text or not intent or not category:
                    stats["blank_rows"] += 1
                    continue
                text_fingerprint = fingerprint(text)
                if text_fingerprint in built_in_exclusions:
                    stats["built_in_excluded_rows"] += 1
                    continue
                if text_fingerprint in optional_exclusions:
                    stats["optional_excluded_rows"] += 1
                    continue
                normalized = normalized_visible(text)
                target = f"{source['domain']}::{intent}"
                before = db.total_changes
                db.execute("INSERT OR IGNORE INTO records VALUES(?,?,?,?,?,?,?,?)",
                    (source["domain"], normalized, target, text, category, source["repo"],
                     source["revision"], source_row))
                stats["duplicates"] += db.total_changes == before
        db.commit()
    db.execute("""CREATE TEMP TABLE conflicts AS SELECT domain,normalized FROM records
        GROUP BY domain,normalized HAVING COUNT(DISTINCT target)>1""")
    stats["conflicting_groups"] = db.execute("SELECT COUNT(*) FROM conflicts").fetchone()[0]
    stats["conflicting_rows"] = db.execute(
        "SELECT COUNT(*) FROM records JOIN conflicts USING(domain,normalized)").fetchone()[0]
    db.execute("DELETE FROM records WHERE EXISTS(SELECT 1 FROM conflicts c "
               "WHERE c.domain=records.domain AND c.normalized=records.normalized)")
    db.commit()
    stats["clean_rows"] = db.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    return stats


def catalog(db: sqlite3.Connection) -> list[dict]:
    result = []
    for target, domain, count in db.execute(
        "SELECT target,domain,COUNT(*) FROM records GROUP BY target,domain ORDER BY domain,target"):
        label = target.split("::", 1)[1]
        result.append({"id": target, "domain": domain, "source_label": label,
            "description": f"{domain.replace('_', ' ')} — {' '.join(label.replace('-', '_').split('_'))}",
            "examples": count})
    return result


def select_catalog(entries: list[dict], candidate_count: int | None) -> list[dict]:
    if candidate_count is None:
        return entries
    if not 2 <= candidate_count <= len(entries):
        raise ValueError(f"Requested {candidate_count} candidates from {len(entries)} available labels")
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for item in entries:
        by_domain[item["domain"]].append(item)
    selected: list[dict] = []
    offset = 0
    while len(selected) < candidate_count:
        for domain in sorted(by_domain):
            if offset < len(by_domain[domain]) and len(selected) < candidate_count:
                selected.append(by_domain[domain][offset])
        offset += 1
    return selected


def emit(db: sqlite3.Connection, output: Path, selected: list[dict], max_rows: int,
         max_per_intent: int) -> tuple[int, str]:
    allowed = {item["id"] for item in selected}
    criteria = {item["id"]: item["description"] for item in selected}
    count = 0
    by_target: dict[str, int] = defaultdict(int)
    with output.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as stream:
                query = ("SELECT domain,normalized,target,text,category,source_repo,source_revision,source_row "
                         "FROM records ORDER BY domain,normalized,target")
                for domain, normalized, target, text, category, repo, revision, source_row in db.execute(query):
                    if target not in allowed or (max_per_intent and by_target[target] >= max_per_intent):
                        continue
                    visible_hash = hashlib.sha256(f"{domain}\0{normalized}".encode()).hexdigest()
                    row = {"id": f"bitext-train:{len(selected)}:{visible_hash[:24]}",
                        "group_id": f"bitext-visible:{visible_hash}", "partition": "train",
                        "state": {"context": {"service_domain": domain}, "utterance": text},
                        "type": "choice",
                        "question": "Select the exact intent for the visible service domain and user request.",
                        "criteria": criteria, "target": target,
                        "task": f"bitext-vertical-{len(selected)}way", "language": "en",
                        "source": repo, "source_revision": revision, "source_row": source_row,
                        "source_category": category, "label_origin": "publisher_train_intent"}
                    stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    count += 1
                    by_target[target] += 1
                    if max_rows and count >= max_rows:
                        break
    return count, sha256(output)


def build(recipe_path: Path, output: Path, source_dir: Path | None = None,
          allow_download: bool = False, candidate_count: int | None = None,
          heldout_path: Path | None = None, max_rows: int = 0,
          max_per_intent: int = 0, license_path: Path = LICENSE_TEXT) -> dict:
    if output.exists():
        raise FileExistsError(f"Output path already exists: {output}")
    if max_rows < 0 or max_per_intent < 0:
        raise ValueError("row limits must be nonnegative")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    if recipe.get("purpose") != "publisher-designated training repositories only":
        raise ValueError("Recipe does not declare a train-only purpose")
    if sha256(license_path) != recipe["license"]["sha256"]:
        raise ValueError("Bundled CDLA license text hash mismatch")
    built_in_document = recipe.get("excluded_training_fingerprints", {})
    built_in = built_in_document.get("sha256", [])
    if not isinstance(built_in, list) or any(
        not isinstance(item, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", item)
        for item in built_in
    ):
        raise ValueError("Recipe training exclusions must be SHA-256 strings")
    built_in = {item.casefold() for item in built_in}
    heldout, heldout_sha = load_fingerprints(heldout_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))
    try:
        db = initialize_db(temporary / "audit.sqlite")
        try:
            stats = ingest(db, recipe["sources"], source_dir, temporary / "downloads",
                           allow_download, built_in, heldout)
            full_catalog = catalog(db)
            selected = select_catalog(full_catalog, candidate_count)
            rows, data_sha = emit(db, temporary / "train.jsonl.gz", selected, max_rows, max_per_intent)
        finally:
            db.close()
        descriptions = {item["id"]: item["description"] for item in selected}
        (temporary / "catalog.json").write_text(
            json.dumps(descriptions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest = {"schema_version": 1, "partition": "train",
            "source_policy": "only pinned publisher-designated training repositories",
            "evaluation_data_fetched": False,
            "training_exclusions": {
                "built_in": {"fingerprints": len(built_in),
                    "rows_removed": stats["built_in_excluded_rows"],
                    "scope": built_in_document.get("scope"),
                    "evaluation_text_packaged": False},
                "optional": None if heldout_sha is None else {
                    "file_sha256": heldout_sha, "fingerprints": len(heldout),
                    "rows_removed": stats["optional_excluded_rows"],
                    "content_packaged": False}},
            "license": recipe["license"],
            "sources": [{key: source[key] for key in
                ("domain", "repo", "revision", "file", "bytes", "sha256")}
                for source in recipe["sources"]],
            "cleaning": stats,
            "catalog": {"available_labels": len(full_catalog), "selected_labels": len(selected),
                "domain_visible_in_every_state": True,
                "selection": "all labels" if candidate_count is None else "deterministic domain round-robin"},
            "output": {"rows": rows, "file": "train.jsonl.gz", "sha256": data_sha},
            "limits": {"max_rows": max_rows, "max_per_intent": max_per_intent}}
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        source_lines = "\n".join(f"- {source['repo']} @ {source['revision']}" for source in recipe["sources"])
        (temporary / "NOTICE.md").write_text(
            "# Bitext training-data notice\n\nDerived from the pinned Bitext training repositories below. "
            "Responses, tags, and system prompts are excluded. The derived data is provided under "
            f"CDLA-Sharing-1.0.\n\n{source_lines}\n", encoding="utf-8")
        shutil.copyfile(license_path, temporary / "LICENSE-DATA.txt")
        for transient in ("audit.sqlite", "downloads"):
            path = temporary / transient
            if path.is_dir(): shutil.rmtree(path)
            elif path.exists(): path.unlink()
        os.replace(temporary, output)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-dir", type=Path)
    source.add_argument("--download", action="store_true")
    parser.add_argument("--candidates", choices=("100", "200", "300", "all"), default="all")
    parser.add_argument("--heldout-fingerprints", type=Path)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-per-intent", type=int, default=100,
                        help="Rows per selected intent; 0 emits every available row")
    args = parser.parse_args(argv)
    count = None if args.candidates == "all" else int(args.candidates)
    result = build(RECIPE, args.out, args.source_dir, args.download, count,
                   args.heldout_fingerprints, args.max_rows, args.max_per_intent)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
