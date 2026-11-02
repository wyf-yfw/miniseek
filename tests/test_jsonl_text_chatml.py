"""JsonlTextDataset accepts SFT conversations as full ChatML text."""
from __future__ import annotations

import json
from pathlib import Path

from algo.train_pretrain import JsonlTextDataset, format_chat_text, jsonl_obj_to_text


def test_jsonl_obj_to_text_prefers_text_field():
    assert jsonl_obj_to_text({"text": "hello", "conversations": []}) == "hello"
    assert jsonl_obj_to_text({"content": "c"}) == "c"


def test_jsonl_obj_to_text_conversations_full_chatml():
    row = {
        "conversations": [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "嗨"},
        ]
    }
    text = jsonl_obj_to_text(row)
    assert text == format_chat_text(row["conversations"])
    assert "<|im_start|>user\n你好<|im_end|>" in text
    assert "<|im_start|>assistant\n嗨<|im_end|>" in text


def test_jsonl_text_dataset_loads_sft_conversations(tmp_path: Path):
    path = tmp_path / "sft.jsonl"
    rows = [
        {
            "conversations": [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
            ]
        },
        {"text": "plain pt line"},
    ]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    ds = JsonlTextDataset(str(path), tokenizer=None, max_samples=None, seed=0)
    assert len(ds.samples) == 2
    assert ds.samples[0].startswith("<|im_start|>user")
    assert "Q1" in ds.samples[0] and "A1" in ds.samples[0]
    assert ds.samples[1] == "plain pt line"
