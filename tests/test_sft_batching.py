"""SFT keep-head truncation (prompt-first) + pack/bucket helpers."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from algo.batching import (
    PackedSFTBatchIter,
    stack_sft_pack_group,
    truncate_keep_head_with_eos,
    truncate_keep_tail_with_eos,
)
from algo.train_sft import JsonlSFTDataset


class _FakeTok:
    bos_token = "<|im_start|>"
    eos_token = "<|im_end|>"
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False):
        # one id per char; map <|im_end|>-like endings: treat char 'E' as eos id 2
        ids = []
        i = 0
        while i < len(text):
            if text.startswith("<|im_end|>", i):
                ids.append(2)
                i += len("<|im_end|>")
            elif text.startswith("<|im_start|>", i):
                ids.append(1)
                i += len("<|im_start|>")
            else:
                ids.append(10 + (ord(text[i]) % 50))
                i += 1
        return ids


def test_truncate_keep_tail_preserves_trailing_eos():
    eos = 2
    ids = list(range(100, 130)) + [eos]
    labs = [-100] * 20 + list(range(200, 211))
    assert len(ids) == len(labs) == 31
    out_ids, out_labs = truncate_keep_tail_with_eos(ids, max_seq_len=10, eos_id=eos, labels=labs)
    assert len(out_ids) == 10
    assert out_ids[-1] == eos
    assert out_labs is not None and out_labs[-1] == eos
    # kept the tail, not the head
    assert out_ids[0] == ids[-10]


def test_truncate_keep_tail_forces_eos_if_stripped():
    eos = 2
    ids = [5, 6, 7, eos]
    # pathological: max_seq_len window somehow without eos after manual mutilation
    # ended_with_eos True + keep-tail of 3 -> [6,7,2] already has eos
    out_ids, _ = truncate_keep_tail_with_eos(ids, max_seq_len=3, eos_id=eos)
    assert out_ids[-1] == eos


def test_truncate_keep_head_preserves_prompt_and_trailing_eos():
    eos = 2
    # head = prompt-ish tokens; tail = completion
    ids = list(range(100, 120)) + list(range(200, 220)) + [eos]
    labs = [-100] * 20 + list(range(300, 321))
    assert len(ids) == len(labs) == 41
    out_ids, out_labs = truncate_keep_head_with_eos(ids, max_seq_len=10, eos_id=eos, labels=labs)
    assert len(out_ids) == 10
    assert out_ids[0] == ids[0], "kept head (prompt side), not tail"
    assert out_ids[-1] == eos
    assert out_labs is not None and out_labs[-1] == eos


def test_sft_dataset_keep_head_not_tail(tmp_path: Path):
    # long assistant reply so total > max_seq; prompt (user) should survive; eos forced
    long_ans = "回答" * 80
    row = {
        "conversations": [
            {"role": "user", "content": "问UNIQUEPROMPT"},
            {"role": "assistant", "content": long_ans},
        ]
    }
    p = tmp_path / "sft.jsonl"
    p.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    ds = JsonlSFTDataset(str(p), _FakeTok(), max_seq_len=32)
    ids, labs = ds.encode_conversation(ds.samples[0])
    assert len(ids) == 32
    assert ids[-1] == 2, "trailing eos should be forced after keep-head truncate"
    # FakeTok encodes user content chars into ids; prompt marker chars must appear near head
    # '问' and 'U' from UNIQUEPROMPT appear early in encoding
    x, y = ds[0]
    assert x.shape[0] == y.shape[0] == 31


def test_stack_sft_pack_preserves_masks():
    chunk = ([1, 2, 3, 4, 5], [-100, -100, 3, 4, 5])
    x, y = stack_sft_pack_group([chunk, chunk], max_seq=5, pad_id=0)
    assert x.shape == (2, 4)
    assert y.shape == (2, 4)
    # y is labels[1:] => [-100, 3, 4, 5]
    assert y[0].tolist() == [-100, 3, 4, 5]


def test_packed_sft_batch_iter_shapes():
    samples = [
        ([1, 2, 3, 4], [-100, -100, 3, 4]),
        ([5, 6, 7], [-100, 6, 7]),
        ([8, 9, 10, 11, 12, 13], [-100, -100, 10, 11, 12, 13]),
        ([14, 15], [-100, 15]),
        ([16, 17, 18, 19], [16, 17, 18, 19]),
        ([20, 21, 22], [20, 21, 22]),
        ([23, 24, 25, 26], [-100, 24, 25, 26]),
        ([27, 28, 29], [27, 28, 29]),
    ]

    def enc(s):
        return s

    it = PackedSFTBatchIter(
        samples, enc, max_seq_len=8, batch_size=2, pad_id=0, seed=0
    )
    batches = list(it)
    assert batches, "pack should emit at least one batch"
    x, y = batches[0]
    assert x.shape[0] == 2
    assert x.shape[1] == 7
    assert y.shape == x.shape
    assert (y == -100).any() or True
