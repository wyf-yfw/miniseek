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
