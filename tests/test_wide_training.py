"""Splitting the pair encoder must preserve full-set scores and training gradients."""
import copy

import torch
from torch.nn import functional as F

from nanodecision2.config import ModelConfig
from nanodecision2.model import DecisionModel


def test_wide_backward_matches_full_candidate_competition():
    torch.manual_seed(17)
    cfg = ModelConfig(vocab_size=32, hidden_size=8, num_attention_heads=2,
                      intermediate_size=12, num_hidden_layers=1, decision_dim=8)
    model = DecisionModel(cfg)
    model.head.dropout.p = 0
    reference = copy.deepcopy(model)
    state = torch.tensor([[2, 4, 6, 1]])
    valid = torch.ones_like(state, dtype=torch.bool)
    ids = torch.randint(1, 32, (1, 1000, 4))
    mask = torch.ones_like(ids, dtype=torch.bool)
    mask[:, ::3, -1] = False
    options = torch.ones(1, 1000, dtype=torch.bool)
    options[:, -3:] = False

    actual = model(state, valid, ids, mask, options)
    context = reference.encoder(state, valid)
    candidates = reference._encode_candidates(ids[0], mask[0])
    expected = reference.head(reference.head.key(context), reference.head.value(context),
                              valid, candidates.unsqueeze(0), options)
    torch.testing.assert_close(actual, expected)
    target = torch.tensor([877])
    F.cross_entropy(expected, target).backward()
    F.cross_entropy(actual, target).backward()
    for (name, parameter), (_, ref) in zip(model.named_parameters(), reference.named_parameters()):
        assert parameter.grad is not None, name
        torch.testing.assert_close(parameter.grad, ref.grad, atol=1e-5, rtol=1e-4)
