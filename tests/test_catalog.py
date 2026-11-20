import json

import pytest

from nanodecision2 import catalog


class FakeEngine:
    def __init__(self, legacy_policy=False):
        self.call = None
        self.legacy_policy = legacy_policy

    def decide(self, state, questions, min_confidence=0, language=None):
        self.call = (state, questions, min_confidence, language)
        ids = list(questions["catalog"]["criteria"])
        probability = 1 / len(ids)
        return {
            "api_contract_version": "test",
            "answers": {"catalog": {
                "selected_option_id": ids[0],
                "value": ids[0],
                "abstained": False,
                "abstain_reasons": [],
                "probabilities": {key: probability for key in ids},
                "raw_probabilities": {key: probability for key in ids},
                "temperature_used": 1.0,
                "calibrated": True,
                "calibration_status": "calibrated",
                "policy_result": {
                    "mode": "legacy" if self.legacy_policy else "advisory",
                    "review_recommended": False,
                    "reasons": [],
                },
            }},
            "model_sha256": "fixture",
        }


def request(size=1000):
    criteria = {
        f"candidate-{index:04d}": f"community service catalog entry {index:04d}"
        for index in range(size)
    }
    criteria["translate-poster"] = "translate a bakery poster into Korean"
    return {
        "state": {"account": "bakery", "visible_policy": "Korean output is supported"},
        "utterance": "Please translate our poster into Korean",
        "question": "Select the requested production capability.",
        "criteria": criteria,
        "language": "en",
    }


def test_thousand_candidate_catalog_is_gold_blind_and_model_gets_grounded_state():
    engine = FakeEngine()
    result = catalog.decide_catalog(engine, request(), top_k=32, min_confidence=.2)
    state, questions, threshold, language = engine.call
    assert state == {
        "context": {"account": "bakery", "visible_policy": "Korean output is supported"},
        "utterance": "Please translate our poster into Korean",
    }
    assert threshold == .2 and language == "en"
    assert len(questions["catalog"]["criteria"]) == 32
    assert "translate-poster" in questions["catalog"]["criteria"]
    assert result["retrieval"]["catalog_size"] == 1001
    assert result["retrieval"]["shortlist_size"] == 32
    answer = result["answers"]["catalog"]
    assert answer["probability_scope"] == "conditional_on_retrieved_shortlist"
    assert not answer["full_catalog_probability_available"]
    assert not answer["calibrated"]
    assert answer["temperature_calibrated"]
    assert answer["temperature_calibration_status"] == "calibrated"
    assert answer["policy_result"]["review_recommended"]
    assert "retrieval_coverage_uncalibrated" in answer["policy_result"]["reasons"]


def test_retrieval_uses_visible_state_and_utterance_and_breaks_ties_by_id():
    criteria = {
        "a": "museum room reservation",
        "b": "bakery poster translation",
        "c": "library membership renewal",
    }
    state_rank = catalog.rank_catalog("The account is a museum", "please help", criteria)
    utterance_rank = catalog.rank_catalog("No account detail", "translate the bakery poster", criteria)
    assert state_rank[0][0] == "a"
    assert utterance_rank[0][0] == "b"
    korean_rank = catalog.rank_catalog(
        "빵집 계정", "포스터를 한국어로 번역해 줘",
        {"translate": "빵집 포스터 한국어 번역", "reserve": "박물관 공간 예약"},
    )
    assert korean_rank[0][0] == "translate"
    short_korean = catalog.rank_catalog(
        "", "빵", {"bread": "빵 굽기", "room": "방 예약"},
    )
    assert short_korean[0][0] == "bread" and short_korean[0][1] > 0
    ties = catalog.rank_catalog("unseen", "vocabulary", {"z": "first option", "a": "second option"})
    assert [key for key, _ in ties] == ["a", "z"]


@pytest.mark.parametrize("payload,match", [
    ({"state": {}, "utterance": "x", "question": "q",
      "criteria": {"a": "same", "b": "same"}}, "Duplicate catalog descriptions"),
    ({"state": {}, "utterance": "x", "question": "q",
      "criteria": {"a": "Same", "b": "ＳＡＭＥ"}}, "Duplicate catalog descriptions"),
    ({"state": {}, "utterance": "x", "question": "q",
      "criteria": {"a": "one", "b": "two"}, "target": "a"}, "Unexpected catalog request"),
    ({"state": {}, "utterance": "", "question": "q",
      "criteria": {"a": "one", "b": "two"}}, "nonempty utterance"),
])
def test_full_catalog_validation_rejects_ambiguous_or_leaking_input(payload, match):
    with pytest.raises(ValueError, match=match):
        catalog.decide_catalog(FakeEngine(), payload)


def test_top_k_contract_and_small_catalog():
    payload = {"state": "visible", "utterance": "pick one", "question": "q",
               "criteria": {"a": "first", "b": "second"}}
    result = catalog.decide_catalog(FakeEngine(), payload, top_k=32)
    assert result["retrieval"]["shortlist_size"] == 2
    answer = result["answers"]["catalog"]
    assert answer["probability_scope"] == "full_catalog"
    assert answer["full_catalog_probability_available"]
    assert not answer["retrieval_coverage_calibrated"]
    assert answer["calibrated"] and answer["calibration_status"] == "calibrated"
    assert not answer["policy_result"]["review_recommended"]
    with pytest.raises(ValueError, match="256"):
        catalog.decide_catalog(FakeEngine(), payload, top_k=1000)


def test_cli_rejects_duplicate_json_keys_before_engine_load(tmp_path, monkeypatch, capsys):
    source = tmp_path / "duplicate.json"
    source.write_text(
        '{"state":{},"utterance":"x","question":"q",'
        '"criteria":{"same":"first","same":"second"}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(catalog, "DecisionEngine",
                        lambda *args, **kwargs: pytest.fail("engine must not load"))
    with pytest.raises(SystemExit) as stopped:
        catalog.main(["--bundle", "unused", "--input", str(source), "--allow-smoke"])
    assert stopped.value.code == 2
    assert "Duplicate JSON object key" in capsys.readouterr().err


def test_cli_forwards_runtime_flags_and_emits_catalog_result(tmp_path, monkeypatch, capsys):
    payload = request(2)
    source = tmp_path / "catalog.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    created = {}

    def engine(bundle, **kwargs):
        created.update(bundle=bundle, **kwargs)
        return FakeEngine()

    monkeypatch.setattr(catalog, "DecisionEngine", engine)
    catalog.main([
        "--bundle", "fixture", "--input", str(source), "--top-k", "2",
        "--min-confidence", ".25", "--legacy-policy", "--allow-uncalibrated",
        "--allow-smoke",
    ])
    output = json.loads(capsys.readouterr().out)
    assert created == {
        "bundle": "fixture", "require_calibration": False,
        "legacy_policy": True, "allow_smoke": True,
    }
    assert output["retrieval"]["shortlist_size"] == 2


def test_root_cli_dispatches_predict_catalog(monkeypatch):
    from nanodecision2 import cli
    seen = []
    monkeypatch.setattr(catalog, "main", lambda argv: seen.append(argv))
    cli.main(["predict-catalog", "--bundle", "fixture", "--input", "request.json"])
    assert seen == [["--bundle", "fixture", "--input", "request.json"]]


def test_zero_score_boundary_ties_force_policy_review_and_legacy_abstention():
    payload = {
        "state": "unseen context",
        "utterance": "unseen request",
        "question": "choose",
        "criteria": {f"id-{index:02d}": f"candidate {index:02d}" for index in range(40)},
    }
    advisory = catalog.decide_catalog(FakeEngine(), payload, top_k=2, min_confidence=.5)
    answer = advisory["answers"]["catalog"]
    assert advisory["retrieval"]["boundary_score"] == 0
    assert advisory["retrieval"]["boundary_tie_count"] == 40
    assert answer["selected_option_id"] in payload["criteria"]
    assert answer["policy_result"] == {
        "mode": "advisory",
        "review_recommended": True,
        "reasons": ["retrieval_coverage_uncalibrated"],
    }
    assert not answer["abstained"]

    legacy = catalog.decide_catalog(FakeEngine(legacy_policy=True), payload, top_k=2)
    legacy_answer = legacy["answers"]["catalog"]
    assert legacy_answer["selected_option_id"] in payload["criteria"]
    assert legacy_answer["value"] is None
    assert legacy_answer["abstained"]
    assert "retrieval_coverage_uncalibrated" in legacy_answer["abstain_reasons"]
