import math
import json

import pytest
import torch

from nanodecision2 import _native
from nanodecision2.config import ModelConfig
from nanodecision2.data import collate, prepare_cache
from nanodecision2.export import export_bundle, write_native
from nanodecision2.model import DecisionModel
from nanodecision2.probability import softmax
from nanodecision2.runtime import DecisionEngine
from nanodecision2.schema import MAX_CHOICE_OPTIONS, normalize_question


def choice(count):
    return {
        "type": "choice",
        "question": "Select one exact capability.",
        "criteria": {f"option-{index}": f"Capability {index}" for index in range(count)},
    }


@pytest.mark.parametrize("count", [2, 256, 257, 1000])
def test_choice_schema_accepts_full_supported_width(count):
    assert len(normalize_question(choice(count))["options"]) == count


def test_schema_keeps_ordered_scale_limit_and_rejects_overflow():
    with pytest.raises(ValueError, match="2..1000 choice options"):
        normalize_question(choice(MAX_CHOICE_OPTIONS + 1))
    levels = [{"value": index, "description": f"level {index}"} for index in range(257)]
    with pytest.raises(ValueError, match="2..256 score options"):
        normalize_question({"type": "score", "question": "Rate it.", "levels": levels})


def test_prepare_cache_accepts_1000_way_choice(tmp_path):
    source = tmp_path / "init"
    source.mkdir()
    (source / "init.json").write_text(json.dumps({"smoke_only": True}), encoding="utf-8")
    (source / "config.json").write_text(json.dumps(ModelConfig.smoke().to_dict()), encoding="utf-8")
    (source / "tokenizer.json").write_text("{}", encoding="utf-8")
    row = {
        "id": "wide-cache",
        "group_id": "wide-cache-group",
        "partition": "train",
        "state": "The visible account and current request are available.",
        **choice(1000),
        "target": "option-999",
    }
    data = tmp_path / "wide.jsonl"
    data.write_text(json.dumps(row) + "\n", encoding="utf-8")
    result = prepare_cache([data], tmp_path / "cache", source, max_state=128, max_pair=128)
    assert result["counts"]["train"] == 1
    assert result["duplicates_removed"] == 0


@pytest.mark.parametrize("count", [257, 1000])
def test_native_scores_one_exact_global_candidate_set(count, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(23)
    model = DecisionModel(ModelConfig.smoke()).eval()
    row = {
        "id": "wide",
        "group_id": "wide-group",
        "state_ids": [2, 31, 32, 1],
        "pair_ids": [[2, 5 + index % 240, 5 + (index * 7) % 240, 1] for index in range(count)],
        "type": "choice",
        "target_probs": [float(index == 0) for index in range(count)],
        "option_ids": [str(index) for index in range(count)],
        "values": None,
        "weight": 1.0,
    }
    batch = collate([row])
    with torch.inference_mode():
        expected = model(**{
            key: batch[key]
            for key in ("state_ids", "state_mask", "pair_ids", "pair_mask", "option_mask", "state_index")
        })[0]
    path = tmp_path / "wide.nd2"
    write_native(model, path, False)
    handle = _native.load(str(path))
    actual = torch.tensor(_native.score(handle, _native.prepare(handle, row["state_ids"]), row["pair_ids"]))
    assert actual.shape == (count,)
    assert torch.allclose(expected, actual, atol=4e-6, rtol=2e-5)
    probabilities = softmax(actual.tolist())
    assert len(probabilities) == count
    assert sum(probabilities) == pytest.approx(1.0, abs=1e-12)
    assert all(math.isfinite(value) and value > 0 for value in probabilities)


def test_native_rejects_more_than_1000_candidates(tmp_path):
    model = DecisionModel(ModelConfig.smoke()).eval()
    path = tmp_path / "wide.nd2"
    write_native(model, path, False)
    handle = _native.load(str(path))
    context = _native.prepare(handle, [2, 31, 1])
    with pytest.raises(RuntimeError, match="2..1000 candidates"):
        _native.score(handle, context, [[2, 5 + index % 240, 1] for index in range(1001)])


def test_wide_decisions_do_not_retain_the_full_tokenized_list(tmp_path):
    model = DecisionModel(ModelConfig.smoke()).eval()
    export_bundle(model, tmp_path, smoke_only=True)
    engine = DecisionEngine(tmp_path, allow_smoke=True)
    context = engine.prepare("Visible account and request.")
    engine.score_prepared(context, choice(2))
    before = engine._pair_cache.cache_info()
    answer = engine.score_prepared(context, choice(257))
    assert len(answer["probabilities"]) == 257
    assert engine._pair_cache.cache_info() == before
