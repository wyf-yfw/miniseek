import random

import torch

from algo.batching import PackedSFTBatchIter, PackedTextBatchIter


class CharTok:
    eos_token_id = 2

    def encode(self, text, add_special_tokens=True):
        return [3 + (ord(c) % 40) for c in text]


def _texts(n=40, seed=0):
    rng = random.Random(seed)
    return ["".join(rng.choice("abcdefghij") for _ in range(rng.randint(1, 30))) for _ in range(n)]


def _batches(**kw):
    it = PackedTextBatchIter(_texts(), CharTok(), max_seq_len=32, batch_size=2, doc_align=2, with_doc_ids=True, **kw)
    return list(it)


def test_docs_start_even():
    batches = _batches()
    assert batches and all(len(b) == 3 for b in batches)
    for x, y, d in batches:
        assert x.shape == y.shape == d.shape == (2, 31)
        starts = (d[:, 1:] != d[:, :-1]).nonzero()
        assert torch.all((starts[:, 1] + 1) % 2 == 0)
        assert torch.all(d[:, 0] == 0)


def test_doc_ids_and_cross_doc_labels():
    for x, y, d in _batches():
        same = d[:, 1:] == d[:, :-1]
        assert torch.all(y[:, :-1][~same] == -100)
        # 文档内：y[t] = x[t+1]，除非 t+1 是补齐用的 pad
        inner = same & (y[:, :-1] != -100)
        assert torch.equal(y[:, :-1][inner], x[:, 1:][inner])
        assert (y == 2).any()  # eos 作为停止符参与训练


def test_sft_pack_doc_ids():
    samples = [([5, 6, 7, 8, 2], [-100, -100, 7, 8, 2]), ([9, 10, 11, 2], [-100, 10, 11, 2])] * 10
    it = PackedSFTBatchIter(samples, lambda s: s, max_seq_len=16, batch_size=2, doc_align=2, with_doc_ids=True)
    batches = list(it)
    assert batches
    for x, y, d in batches:
        starts = (d[:, 1:] != d[:, :-1]).nonzero()
        assert torch.all((starts[:, 1] + 1) % 2 == 0)
        same = d[:, 1:] == d[:, :-1]
        assert torch.all(y[:, :-1][~same] == -100)
        assert torch.all(y[x == 5] == -100)  # prompt 首 token 预测的仍是 prompt，不算损失


def test_default_behavior_unchanged():
    it = PackedTextBatchIter(_texts(), CharTok(), max_seq_len=32, batch_size=2)
    first = next(iter(it))
    assert len(first) == 2 and first[0].shape == (2, 31)
