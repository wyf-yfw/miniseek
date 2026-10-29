"""--max-samples 先从全量均匀抽取，再截断，不用文件前缀。"""
from __future__ import annotations

import json
from pathlib import Path

from algo.train_pretrain import JsonlTextDataset, reservoir_subset
from algo.train_sft import JsonlSFTDataset


def _sft_row(tag: str) -> str:
    row = {
        "conversations": [
            {"role": "user", "content": tag},
            {"role": "assistant", "content": "ok"},
        ]
    }
    return json.dumps(row, ensure_ascii=False)


def _write_blocked(path: Path, head: int, tail: int) -> None:
    lines = [_sft_row(f"A{i}") for i in range(head)]
    lines += [_sft_row(f"B{i}") for i in range(tail)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _tags(ds: JsonlSFTDataset) -> list[str]:
    return [conv[0]["content"] for conv in ds.samples]


def test_reservoir_is_not_a_prefix():
    kept, n = reservoir_subset(list(range(100)), 5, seed=0)
    assert n == 100
    assert len(kept) == 5
    assert kept != [0, 1, 2, 3, 4]
    assert len(set(kept)) == 5


def test_reservoir_keeps_order_when_not_truncating():
    kept, n = reservoir_subset(["a", "b", "c"], None, seed=1)
    assert (kept, n) == (["a", "b", "c"], 3)
    kept, n = reservoir_subset(["a", "b", "c"], 10, seed=1)
    assert (kept, n) == (["a", "b", "c"], 3)


def test_sft_max_samples_reaches_tail_block(tmp_path: Path):
    path = tmp_path / "sft.jsonl"
    _write_blocked(path, head=20, tail=10)
    ds = JsonlSFTDataset(str(path), tokenizer=None, max_samples=5, seed=0)
    tags = _tags(ds)
    assert ds.n_seen == 30
    assert len(tags) == 5
    assert tags != [f"A{i}" for i in range(5)]
    again = _tags(JsonlSFTDataset(str(path), tokenizer=None, max_samples=5, seed=0))
    assert again == tags

    seen = set()
    for seed in range(40):
        seen.update(_tags(JsonlSFTDataset(str(path), tokenizer=None, max_samples=5, seed=seed)))
    assert any(tag.startswith("B") for tag in seen)

    full = JsonlSFTDataset(str(path), tokenizer=None)
    assert _tags(full) == [f"A{i}" for i in range(20)] + [f"B{i}" for i in range(10)]
    assert full.n_seen == 30


def test_pretrain_max_samples_reaches_tail_block(tmp_path: Path):
    path = tmp_path / "pt.jsonl"
    lines = [json.dumps({"text": f"A{i}"}, ensure_ascii=False) for i in range(20)]
    lines += [json.dumps({"text": f"B{i}"}, ensure_ascii=False) for i in range(10)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ds = JsonlTextDataset(str(path), tokenizer=None, max_samples=5, seed=0)
    assert ds.n_seen == 30
    assert ds.samples != [f"A{i}" for i in range(5)]
    seen = set()
    for seed in range(40):
        seen.update(JsonlTextDataset(str(path), tokenizer=None, max_samples=5, seed=seed).samples)
    assert any(text.startswith("B") for text in seen)
