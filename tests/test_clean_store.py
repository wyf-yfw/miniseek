from __future__ import annotations

from api import clean_store as cs


def test_create_and_list(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "CLEAN_ROOT", tmp_path / "clean")
    row = cs.create_job(name="demo", input_path="data/pretrain_mix_5g.jsonl")
    assert row["status"] == "draft"
    assert row["operators"]["length"]["enabled"] is True
    assert (tmp_path / "clean" / row["id"] / "job.json").is_file()
    assert cs.list_jobs()[0]["id"] == row["id"]


def test_patch_resets_finished(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "CLEAN_ROOT", tmp_path / "clean")
    row = cs.create_job(input_path="a.jsonl")
    row["status"] = "finished"
    cs.save_job(row)
    patched = cs.patch_job(row["id"], input_path="b.jsonl")
    assert patched["status"] == "draft"
    assert patched["input_path"] == "b.jsonl"
