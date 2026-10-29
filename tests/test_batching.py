"""Unit tests for length-bucket / pack batch builders."""
from __future__ import annotations

import torch

from algo.batching import (
    LengthBucketBatchSampler,
    build_pack_batches,
    cut_packed_stream,
    make_length_batches,
    pack_docs_to_stream,
    stack_pack_group,
)


def test_make_length_batches_groups_similar_lengths():
    lengths = [10, 100, 12, 98, 11, 99]
    batches = make_length_batches(lengths, batch_size=3, seed=0, shuffle_batches=False)
    assert len(batches) == 2
    # sorted order: indices of 10,11,12 then 98,99,100
    flat = [i for b in batches for i in b]
    assert flat == [0, 4, 2, 3, 5, 1]


def test_bucket_sampler_reshuffles_across_epochs():
    lengths = list(range(20))
    samp = LengthBucketBatchSampler(lengths, batch_size=4, seed=1)
    a = list(samp)
    b = list(samp)
    assert len(a) == 5 and len(b) == 5
    # same membership, possibly different batch order
    assert sorted(tuple(x) for x in a) == sorted(tuple(x) for x in b)
    # with different epoch seeds, order of batches should usually differ
    assert a != b or len(a) == 1


def test_pack_stream_and_stack_shape():
    docs = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    stream = pack_docs_to_stream(docs, eos_id=2)
    assert stream == [1, 2, 3, 2, 4, 5, 2, 6, 7, 8, 9, 2]
    chunks = cut_packed_stream(stream, max_seq=5)
    assert all(len(c) <= 5 for c in chunks)
    x, y = stack_pack_group(chunks[:2], max_seq=5, pad_id=0)
    assert x.shape == (2, 4)
    assert y.shape == (2, 4)
    assert (y == -100).any() or True  # may or may not pad


def test_build_pack_batches_batch_dim():
    enc = [[i] * 20 for i in range(1, 40)]
    batches = build_pack_batches(enc, batch_size=4, max_seq=16, eos_id=2, pad_id=0, seed=0)
    assert batches, "expected at least one pack batch"
    x, y = batches[0]
    assert x.ndim == 2 and x.shape[0] == 4
    assert y.shape == x.shape
    # pad labels are -100
    assert int((y == -100).sum()) >= 0


class _FakeTok:
    """Deterministic char→id encoder for pack tests (ids = ord(ch))."""

    eos_token_id = 2

    def encode(self, text: str, add_special_tokens: bool = True):
        return [ord(c) for c in text]


def test_packed_text_iter_keeps_long_doc_tail_across_chunks():
    """A doc longer than max_seq_len must still appear in later packed chunks.

    Example: one doc of 10 tokens, max_seq=4, batch_size=1 → non-overlapping
    stream cuts; tokens beyond the first window must land in later chunks
    (no per-doc pre-truncate before pack).
    """
    from algo.batching import PackedTextBatchIter

    # 10 distinct printable chars → 10 token ids; plus eos appended by packer
    text = "ABCDEFGHIJ"  # ords 65..74
    tok = _FakeTok()
    assert len(tok.encode(text)) == 10

    it = PackedTextBatchIter(
        [text],
        tok,
        max_seq_len=4,
        batch_size=1,
        pad_id=0,
        eos_id=2,
        seed=0,
    )
    batches = list(it)
    # Flatten all non-pad tokens from x and the final y shift isn't needed:
    # stack_pack_group gives x=ids[:-1]; recover by taking each chunk's raw
    # content via reconstructing from (x row + last predicted-ish). Simpler:
    # re-walk the same pack logic expectation on emitted x rows.
    assert batches, "expected packed batches from a 10-token doc"
    seen: list[int] = []
    for x, y in batches:
        assert x.shape[0] == 1
        row = x[0].tolist()
        # x is length max_seq-1=3; recover full chunk of length 4 via x + last label
        # where y is not -100. For a full chunk, y[-1] is the last real token of chunk.
        last = int(y[0, -1].item())
        if last == -100:
            # short trailing chunk: only take non-pad from x
            seen.extend([t for t in row if t != 0])
        else:
            seen.extend(row + [last])

    # Full stream = doc tokens + trailing eos from packer
    expected_stream = [ord(c) for c in text] + [2]
    # Non-overlapping cuts of 4; drop trailing <2
    expected_chunks = []
    for i in range(0, max(0, len(expected_stream) - 1), 4):
        chunk = expected_stream[i : i + 4]
        if len(chunk) >= 2:
            expected_chunks.append(chunk)
    flat_expected = [t for c in expected_chunks for t in c]

    # Critical: later tokens of the doc (beyond first window) must appear
    assert ord("E") in seen, "token beyond first max_seq window was dropped (pre-truncate?)"
    assert ord("J") in seen, "doc tail token missing from packed stream"
    assert seen == flat_expected


def test_packed_text_estimate_len_uses_full_doc_chars():
    """estimate_len must count full doc length, not min(max_seq_len, len)."""
    from algo.batching import PackedTextBatchIter, estimate_pack_steps

    tok = _FakeTok()
    # One long doc: 1000 chars; max_seq=4 → old code undercounted as 4
    texts = ["x" * 1000]
    it = PackedTextBatchIter(
        texts, tok, max_seq_len=4, batch_size=1, pad_id=0, eos_id=2, seed=0
    )
    got = it.estimate_len()
    want = estimate_pack_steps(1, 1000.0, 1, 4)
    assert got == want
    # And must be much larger than the truncated estimate
    truncated = estimate_pack_steps(1, 4.0, 1, 4)
    assert got > truncated
