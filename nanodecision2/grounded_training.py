"""Generate train-only grounded choice data with matched candidate widths.

The catalog contains 1,000 operational routes. It is a Cartesian product of
ten service accounts and one hundred artifact/operation capabilities, so it
must not be reported as 1,000 independent human intents.
"""
import argparse
from contextlib import contextmanager
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import random
import secrets

from .runtime import sha256


CLIENTS = (
    ("library", "public library", "공공 도서관"), ("museum", "local museum", "지역 박물관"),
    ("school", "community school", "지역 학교"), ("clinic", "neighborhood clinic", "동네 의원"),
    ("theatre", "independent theatre", "독립 극장"), ("cooperative", "consumer cooperative", "생활 협동조합"),
    ("sports_club", "amateur sports club", "생활 체육 동호회"), ("nonprofit", "local nonprofit", "지역 비영리 단체"),
    ("studio", "small design studio", "소규모 디자인 스튜디오"), ("bakery", "neighborhood bakery", "동네 빵집"),
)
ASSETS = (
    ("poster", "poster", "포스터"), ("brochure", "brochure", "브로슈어"),
    ("invitation", "invitation", "초대장"), ("report", "report", "보고서"),
    ("presentation", "presentation", "발표 자료"), ("social_card", "social media card", "소셜 미디어 카드"),
    ("certificate", "certificate", "인증서"), ("newsletter", "newsletter", "뉴스레터"),
    ("infographic", "infographic", "인포그래픽"), ("digital_sign", "digital sign", "디지털 안내판"),
)
ACTIONS = (
    ("create", "create the {asset}", "{asset} 신규 제작"),
    ("revise_copy", "revise the copy in the {asset}", "{asset} 문구 수정"),
    ("translate", "translate the {asset}", "{asset} 번역"),
    ("replace_image", "replace the image in the {asset}", "{asset} 이미지 교체"),
    ("resize", "resize the {asset}", "{asset} 규격 변경"),
    ("adjust_color", "adjust the colors of the {asset}", "{asset} 색상 조정"),
    ("accessibility", "check the accessibility of the {asset}", "{asset} 접근성 점검"),
    ("prepare_print", "prepare the {asset} for printing", "{asset} 인쇄 준비"),
    ("export_web", "export the {asset} for the web", "{asset} 웹용 내보내기"),
    ("proofread", "proofread the {asset}", "{asset} 교정"),
)
WIDTHS = (32, 100, 200, 300, 1000)


def _seed(seed, *parts):
    value = ":".join(map(str, (seed, *parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def catalog():
    records = []
    for client_id, client_en, client_ko in CLIENTS:
        for asset_id, asset_en, asset_ko in ASSETS:
            for action_id, action_en, action_ko in ACTIONS:
                records.append({
                    "id": f"{client_id}__{asset_id}__{action_id}",
                    "client": client_id, "asset": asset_id, "action": action_id,
                    "client_en": client_en, "client_ko": client_ko,
                    "asset_en": asset_en, "asset_ko": asset_ko,
                    "action_en": action_en.format(asset=asset_en),
                    "action_ko": action_ko.format(asset=asset_ko),
                    "en": f"{client_en} account: {action_en.format(asset=asset_en)}",
                    "ko": f"{client_ko} 계정: {action_ko.format(asset=asset_ko)}",
                })
    assert len(records) == len({item["id"] for item in records}) == 1000
    return records


def contrast_groups():
    """Yield 250 disjoint 2x2 groups covering every catalog route once."""
    requests = [(asset[0], action[0]) for asset in ASSETS for action in ACTIONS]
    request_pairs = [(requests[i], requests[i + 1]) for i in range(0, len(requests), 2)]
    for client_pair in range(5):
        clients = (CLIENTS[client_pair * 2][0], CLIENTS[client_pair * 2 + 1][0])
        for request_pair, paired_requests in enumerate(request_pairs):
            yield f"grounded-routing:{client_pair}:{request_pair}", clients, paired_requests


def near_neighbors(gold, records):
    axes = (
        [x["id"] for x in records if x["client"] == gold["client"] and x["asset"] == gold["asset"] and x["action"] != gold["action"]],
        [x["id"] for x in records if x["client"] == gold["client"] and x["action"] == gold["action"] and x["asset"] != gold["asset"]],
        [x["id"] for x in records if x["asset"] == gold["asset"] and x["action"] == gold["action"] and x["client"] != gold["client"]],
    )
    assert [len(axis) for axis in axes] == [9, 9, 9]
    return [item for axis in axes for item in axis]


def ranked_group_candidates(group_id, golds, records, width, seed):
    """One nested candidate list shared by every cell in a grounding factorial."""
    if width not in WIDTHS:
        raise ValueError(f"width must be one of {WIDTHS}")
    gold_ids = [gold["id"] for gold in golds]
    clients = sorted({gold["client"] for gold in golds})
    assets = {gold["asset"] for gold in golds}
    actions = sorted({gold["action"] for gold in golds})
    if len(clients) != 2 or len(assets) != 1 or len(actions) != 2:
        raise ValueError("A grounding group must be two clients by two requests for one artifact")
    asset = next(iter(assets))
    by_id = {item["id"]: item for item in records}
    representative = list(gold_ids)
    alternate_actions = [item[0] for item in ACTIONS if item[0] not in actions][:3]
    alternate_assets = [item[0] for item in ASSETS if item[0] != asset][:4]
    alternate_clients = [item[0] for item in CLIENTS if item[0] not in clients][:3]
    for client in clients:
        representative.extend(f"{client}__{asset}__{action}" for action in alternate_actions)
    for client in clients:
        for action in actions:
            representative.extend(f"{client}__{other_asset}__{action}" for other_asset in alternate_assets)
    for action in actions:
        representative.extend(f"{other_client}__{asset}__{action}" for other_client in alternate_clients)
    full_near = set(gold_ids)
    for gold in golds:
        near = near_neighbors(gold, records)
        full_near.update(near)
    representative = list(dict.fromkeys(representative))
    assert len(representative) == 32 and set(representative) <= by_id.keys()
    remaining_near = sorted(full_near - set(representative))
    random.Random(_seed(seed, group_id, "remaining-near")).shuffle(remaining_near)
    hard = representative + remaining_near
    used = set(hard)
    far = [item["id"] for item in records if item["id"] not in used]
    random.Random(_seed(seed, group_id, "far")).shuffle(far)
    selected = (hard + far)[:width]
    assert len(selected) == len(set(selected)) == width
    return selected


@contextmanager
def _output_stream(path, compressed):
    if compressed:
        with path.open("xb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=6, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as stream:
                    yield stream
    else:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            yield stream


def _publish_new(source, destination):
    """Atomically publish a file without replacing a concurrent writer."""
    os.link(source, destination)
    source.unlink()


def _utterance(item, language):
    if language == "ko":
        return f"{item['action_ko']} 작업을 진행해 줘."
    return f"Please {item['action_en']}."


def _state(item, language, variant):
    context = {
        "current_service_account": item[f"client_{language}"],
        "routing_status": "active" if language == "en" else "활성",
        "effective_rule": ("Use the current account together with the requested artifact and operation."
                           if language == "en" else "현재 계정과 요청된 제작물 및 작업을 함께 사용한다."),
    }
    if variant:
        context["request_channel"] = "web intake" if language == "en" else "웹 접수"
        context["display_theme"] = "high contrast" if language == "en" else "고대비"
    return {"context": context, "utterance": _utterance(item, language)}


def rows(width, languages=("en", "ko"), limit_groups=None, seed=42):
    records = catalog()
    by_id = {item["id"]: item for item in records}
    groups = list(contrast_groups())
    if limit_groups is not None:
        if type(limit_groups) is not int or not 1 <= limit_groups <= len(groups):
            raise ValueError(f"limit_groups must be in [1, {len(groups)}]")
        groups = groups[:limit_groups]
    for group_id, clients, requests in groups:
        golds = [by_id[f"{client}__{asset}__{action}"]
                 for client in clients for asset, action in requests]
        selected = ranked_group_candidates(group_id, golds, records, width, seed)
        for client in clients:
            for asset, action in requests:
                gold = by_id[f"{client}__{asset}__{action}"]
                for language in languages:
                    descriptions = {item_id: by_id[item_id][language] for item_id in selected}
                    ordered = list(descriptions)
                    random.Random(_seed(seed, group_id, width, language, "display")).shuffle(ordered)
                    criteria = {item_id: descriptions[item_id] for item_id in ordered}
                    for variant in (0, 1):
                        state = _state(gold, language, variant)
                        assert gold["id"] not in json.dumps(state, ensure_ascii=False)
                        yield {
                            "id": f"{group_id}:k{width}:{client}:{asset}:{action}:{language}:{variant}",
                            "group_id": group_id, "semantic_group_id": group_id, "partition": "train",
                            "state": state, "type": "choice",
                            "question": ("Select the exact account-specific production route requested."
                                         if language == "en" else "현재 계정과 요청에 맞는 정확한 제작 경로를 선택하라."),
                            "criteria": criteria,
                            "target": gold["id"], "task": "grounded_creative_routing",
                            "language": language, "source": "nd2-authored-grounded-routing-v1",
                            "label_origin": "deterministic_state_request_oracle", "synthetic": True,
                            "candidate_count": width, "context_variant": variant,
                        }


def generate(out, width, language="both", limit_groups=None, seed=42):
    if language not in ("en", "ko", "both"):
        raise ValueError("language must be en, ko or both")
    out = Path(out)
    manifest = out.with_suffix(out.suffix + ".manifest.json")
    if out.exists() or manifest.exists():
        raise FileExistsError(out if out.exists() else manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    nonce = f"{os.getpid()}-{secrets.token_hex(8)}"
    temp = out.with_name(out.name + f".partial-{nonce}")
    languages = ("en", "ko") if language == "both" else (language,)
    count = 0
    groups = set()
    targets = set()
    manifest_temp = manifest.with_name(manifest.name + f".partial-{nonce}")
    try:
        with _output_stream(temp, out.suffix == ".gz") as stream:
            for row in rows(width, languages, limit_groups, seed):
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
                count += 1
                groups.add(row["group_id"])
                targets.add(row["target"])
        report = {
            "version": 1, "partition": "train", "rows": count, "contrast_groups": len(groups),
            "distinct_route_targets": len(targets), "catalog_routes": 1000,
            "base_artifact_operation_capabilities": 100, "candidate_count": width,
            "languages": list(languages), "seed": seed, "sha256": sha256(temp),
            "compressed": out.suffix == ".gz",
            "group_contract": "Each group is a state-client x request-capability 2x2 factorial and is stable across candidate widths.",
            "candidate_contract": "All four factorial cells and irrelevant-context variants share one ordered list. Widths are nested; K=32 provides four one-axis neighbors per axis and cell, and K>=100 contains the full one-axis union.",
            "scope": "Authored synthetic training data. The 1,000 routes are not 1,000 independent human intents.",
        }
        manifest_temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _publish_new(temp, out)
        try:
            _publish_new(manifest_temp, manifest)
        except Exception:
            if out.exists() and sha256(out) == report["sha256"]:
                out.unlink()
            raise
    finally:
        temp.unlink(missing_ok=True)
        manifest_temp.unlink(missing_ok=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--k", type=int, choices=WIDTHS, required=True)
    parser.add_argument("--language", choices=("en", "ko", "both"), default="both")
    parser.add_argument("--limit-groups", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    print(json.dumps(generate(args.out, args.k, args.language, args.limit_groups, args.seed),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
