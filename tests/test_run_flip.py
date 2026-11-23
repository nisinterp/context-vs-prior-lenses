"""Lightweight tests on dummy data: no model, no lens, no downloads.

Run from the repo root:
    python -m unittest discover tests
"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiment"))
import run_flip as R  # noqa: E402


class DummyTokenizer:
    """Maps every distinct word to its own id; a leading space gives a different id (like BPE)."""

    def __init__(self):
        self.vocab = {}

    def encode(self, text, add_special_tokens=False):
        first = (" " if text.startswith(" ") else "") + text.split()[0]
        return [self.vocab.setdefault(first, len(self.vocab))]


class DummyModel(torch.nn.Module):
    """Just the parts lens_logits uses: model.model.norm and model.lm_head."""

    def __init__(self, d=4, vocab=6):
        super().__init__()
        torch.manual_seed(0)
        self.model = SimpleNamespace(norm=torch.nn.LayerNorm(d))
        self.lm_head = torch.nn.Linear(d, vocab, bias=False)


class TestLayerMetrics(unittest.TestCase):
    def test_flip_layer(self):
        self.assertEqual(R.flip_layer(torch.tensor([-1.0, 0.5, -0.2, 1.0, 2.0])), 3)
        self.assertEqual(R.flip_layer(torch.tensor([0.1, 0.2])), 0)
        self.assertIsNone(R.flip_layer(torch.tensor([1.0, -1.0])))   # last layer negative -> no flip

    def test_divergence_layer(self):
        d = torch.tensor([0.0, 3.0, 0.5, 4.0, 5.0])
        d0 = torch.zeros(5)
        self.assertEqual(R.divergence_layer(d, d0, 2.0), 3)
        self.assertIsNone(R.divergence_layer(d, d0, 10.0))

    def test_peak_prior(self):
        self.assertEqual(R.peak_prior(torch.tensor([0.0, -2.5, -1.0])), (1, -2.5))

    def test_delta_uses_logsumexp_over_spellings(self):
        logits = torch.tensor([[1.0, 1.0, 0.0, 5.0]])
        expected = torch.logsumexp(torch.tensor([1.0, 1.0]), 0) - 0.0
        self.assertAlmostEqual(float(R.delta(logits, [0, 1], [2])[0]), float(expected), places=5)

    def test_ranks(self):
        logits = torch.tensor([[0.1, 0.9, 0.5], [0.9, 0.1, 0.5]])
        self.assertEqual(R.ranks(logits, [1]).tolist(), [0, 2])


class TestPrompts(unittest.TestCase):
    def test_strip_assertion(self):
        prompt = "Paris is the capital of Italy. What is the capital of Italy?"
        self.assertEqual(R.strip_assertion(prompt, "Paris is the capital of Italy."),
                         "What is the capital of Italy?")

    def test_first_token_ids_four_spellings(self):
        tok = DummyTokenizer()
        ids = R.first_token_ids(tok, "red apple")
        # "red", " red", "Red", " Red" -> four different ids
        self.assertEqual(len(ids), 4)
        self.assertEqual(ids, sorted(ids))


class TestLenses(unittest.TestCase):
    def setUp(self):
        self.model = DummyModel()
        self.hs = torch.randn(3, 4)                         # 3 layers, hidden size 4
        self.J = {0: torch.randn(4, 4), 1: torch.randn(4, 4)}  # no matrix for the last layer

    def test_shapes(self):
        ll, jl = R.lens_logits(self.model, self.J, self.hs)
        self.assertEqual(tuple(ll.shape), (3, 6))
        self.assertEqual(sorted(jl), [0, 1])

    def test_selected_tokens_give_same_delta_as_full_vocab(self):
        ids = {"context": [1], "prior": [2, 3], "distractor": [4]}
        sel = sorted(set().union(*ids.values()))
        full = R.curves(*R.lens_logits(self.model, self.J, self.hs), ids)
        part = R.curves(*R.lens_logits(self.model, self.J, self.hs, sel=sel), ids)
        for key in ("ll", "jl", "jl_dtr"):
            self.assertTrue(torch.allclose(full[key], part[key], atol=1e-5), key)
        self.assertEqual(part["jl_layers"], [0, 1])


if __name__ == "__main__":
    unittest.main()
