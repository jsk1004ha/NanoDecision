"""Language-aware training utilities. No extra inference parameters.

Equivalence regularization is opt-in for reviewed translations. Sharing a class
label is NOT sufficient evidence that two arbitrary examples are equivalent.
"""
import math
from collections import defaultdict


def language_key(value):
    if not isinstance(value, str) or not value.strip():
        return "unknown"
    return value.lower().replace("_", "-").split("-")[0]


def probability_bucket(language, kind, count):
    count_bucket = "2" if count == 2 else "3-5" if count <= 5 else "6-10" if count <= 10 else "11-20" if count <= 20 else "21+"
    return f"{language_key(language)}|{kind}|{count_bucket}"


def served_temperature(calibration, language, kind, count):
    if not calibration:
        return 1.0
    table = calibration.get("temperatures_by_bucket", {})
    cell = table.get(probability_bucket(language, kind, count))
    return float(cell["temperature"] if cell else calibration["temperature"])


def validate_language_weights(weights):
    if not isinstance(weights, dict):
        raise ValueError("Language weights must be a JSON object")
    result = {}
    for key, value in weights.items():
        value = float(value)
        if not math.isfinite(value) or not 0 < value <= 100:
            raise ValueError("Language weights must be finite, positive and <= 100")
        key = language_key(key)
        if key in result:
            raise ValueError("Duplicate normalized language key")
        result[key] = value
    return result


def reviewed_pair_units(dataset):
    """Only metadata/indexes stay resident; tokenized examples stay on disk."""
    by_pair = defaultdict(list)
    singles = []
    metadata = {}
    for i in range(len(dataset)):
        row = dataset[i]
        pair = row.get("equivalence_id")
        if not pair or row.get("equivalence_verified") is not True:
            singles.append((i,)); continue
        meta = (row["group_id"], row["type"], tuple(row["values"] or []),
                tuple(sorted(zip(row["option_ids"], row["target_probs"]))))
        if pair in metadata and metadata[pair] != meta:
            raise ValueError("Reviewed equivalents must share group, candidate IDs, gold distribution and score levels")
        metadata[pair] = meta
        by_pair[pair].append((i, language_key(row.get("language"))))
    paired = []
    for entries in by_pair.values():
        if len(entries) != 2 or entries[0][1] == entries[1][1]:
            raise ValueError("Each approved equivalence_id must have exactly two different-language rows")
        paired.append(tuple(i for i, _ in entries))
    if not paired:
        raise ValueError("Pair consistency requested but no reviewed bilingual pairs exist")
    return singles + paired


def bilingual_js(logits, rows, original_order):
    """Mean JS(p_ko, p_en), aligning by candidate ID, not physical position."""
    import torch
    from torch.nn import functional as F
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        if row.get("equivalence_verified") is True and row.get("equivalence_id"):
            groups[row["equivalence_id"]].append(i)
    losses = []
    for members in groups.values():
        if len(members) != 2:
            continue
        i, j = members
        a, b = rows[i], rows[j]
        if a["group_id"] != b["group_id"] or a["type"] != b["type"]:
            raise ValueError("Equivalence group/type conflict")
        ids = sorted(a["option_ids"])
        if ids != sorted(b["option_ids"]) or a["values"] != b["values"]:
            raise ValueError("Equivalence candidates/levels conflict")
        if {k:v for k,v in zip(a["option_ids"],a["target_probs"])} != {k:v for k,v in zip(b["option_ids"],b["target_probs"])}:
            raise ValueError("Equivalent pair gold labels disagree")
        def canonical(k, row):
            order = original_order[k, :len(ids)].detach().cpu().tolist()
            mapped = {row["option_ids"][old]: pos for pos, old in enumerate(order)}
            return F.log_softmax(logits[k, [mapped[c] for c in ids]].float(), dim=-1)
        la, lb = canonical(i, a), canonical(j, b)
        lm = torch.logaddexp(la, lb) - math.log(2)
        losses.append(0.5 * ((la.exp() * (la-lm)).sum() + (lb.exp() * (lb-lm)).sum()))
    return (torch.stack(losses).mean() if losses else logits.sum() * 0.0), len(losses)


def paired_batches(units, lengths, batch_size, seed):
    import numpy as np
    if batch_size < 2:
        raise ValueError("Pair consistency needs micro-batch >= 2")
    order = np.arange(len(units)); np.random.default_rng(seed).shuffle(order)
    # Stable length grouping without ever splitting a bilingual unit.
    for lo in range(0, len(order), 1024):
        part = order[lo:lo+1024]
        scores = [max(int(lengths[i]) for i in units[int(k)]) for k in part]
        order[lo:lo+1024] = part[np.argsort(scores, kind="stable")]
    current = []
    for k in order:
        unit = units[int(k)]
        if current and len(current)+len(unit) > batch_size:
            yield current; current = []
        current.extend(unit)
    if current:
        yield current
