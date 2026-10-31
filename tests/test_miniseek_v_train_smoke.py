import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from algo import miniseek_v_train as mvt
from algo.batching import PackedSFTBatchIter, PackedTextBatchIter
from algo.model.miniseek_v import FlashConfig, Transformer

ROOT = Path(__file__).resolve().parents[1]


class CharTok:
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text, add_special_tokens=True):
        return [3 + (ord(c) % 90) for c in text]


def _model():
    torch.manual_seed(0)
    return Transformer(FlashConfig.tiny())


def test_doc_align_for_f01():
    assert mvt.doc_align(FlashConfig()) == 2


def test_pretrain_steps_with_doc_ids():
    model = _model()
    texts = ["深度学习是机器学习的分支" * 2, "北京是中国的首都", "大语言模型可以生成文本" * 3] * 4
    it = PackedTextBatchIter(texts, CharTok(), max_seq_len=48, batch_size=2, doc_align=2, with_doc_ids=True)
    batches = list(it)
    opt = mvt.build_optimizer(model, 3e-3, fused=False)
    first = last = None
    for _ in range(6):
        for batch in batches[:1]:
            ids, labels, docs = mvt.unpack_batch(batch, torch.device("cpu"))
            loss, ce, extras = mvt.forward_loss(model, ids, labels, docs, aux_coef=1e-4)
            opt.zero_grad()
            loss.backward()
            opt.step()
            model.update_expert_bias()
            first = first if first is not None else float(ce.detach())
            last = float(ce.detach())
    assert last < first


def test_sft_forward_with_replay_and_candidates():
    model = _model()
    samples = [([5, 6, 7, 8, 9, 10, 11, 2], [-100, -100, -100, -100, 9, 10, 11, 2])] * 8
    it = PackedSFTBatchIter(samples, lambda s: s, max_seq_len=32, batch_size=2, doc_align=2, with_doc_ids=True)
    ids, labels, docs = next(iter(it))
    loss, ce, extras = mvt.forward_loss(
        model, ids, labels, docs, aux_coef=1e-4, use_candidate_pool=True, bounded_replay=True, window=8
    )
    assert torch.isfinite(loss) and torch.isfinite(extras["indexer_kl"])
    loss.backward()


def test_ckpt_state_roundtrip():
    model = _model()
    sd = mvt.state_for_ckpt(model)
    assert sd["hasher.token_map"].dtype == torch.long
    again = _model()
    with torch.no_grad():
        for p in again.parameters():
            p.add_(1.0)
    again.load_state_dict(mvt.clean_state_dict({"_orig_mod." + k: v for k, v in sd.items()}))
    for k, v in again.state_dict().items():
        assert torch.equal(v.float(), sd[k].float()), k


def test_metrics_regex_reads_v41_log_line():
    from api.task_runner import STEP_RE

    ce = torch.tensor(2.5)
    extras = {"balance": torch.tensor(1.2), "indexer_kl": torch.tensor(0.03)}
    line = mvt.log_line(0, 7, 2.53, ce, extras, 3e-4)
    m = STEP_RE.search(line)
    assert m and m.group(1) == "7" and m.group(2) == "2.5300" and m.group(4) == "3.00e-04"
    assert STEP_RE.search("epoch=0 step=3 loss=1.0 ce=0.9 aux=1.1 ppl=2.5 lr=1e-4")


def _run(module, args, env):
    proc = subprocess.run(
        [sys.executable, "-m", module, *args], cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8"
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    return proc.stdout


@pytest.mark.slow
def test_cli_pretrain_then_sft(tmp_path):
    env = {
        **os.environ,
        "MINISEEK_M_PRESET": "tiny",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    pt_data = tmp_path / "pt.jsonl"
    pt_data.write_text(
        "\n".join(json.dumps({"text": t * 3}, ensure_ascii=False) for t in ["深度学习是机器学习的分支。", "北京是中国的首都。"] * 6),
        encoding="utf-8",
    )
    pt_out = tmp_path / "pt"
    out = _run(
        "algo.train_pretrain",
        ["--arch", "miniseek-v", "--data", str(pt_data), "--out-dir", str(pt_out), "--max-steps", "3", "--batch-size", "2",
         "--max-seq-len", "32", "--no-compile", "--log-every", "1", "--num-workers", "0"],
        env,
    )
    assert "[miniseek-v]" in out and "MiniSeek-V" in out and "kl=" in out and "expert_load=" in out
    blob = torch.load(pt_out / "last.pt", map_location="cpu", weights_only=False)
    assert blob["config"]["arch"] == "miniseek-v" and set(blob["optimizer"]) == {"muon", "adamw", "sinkhorn"}

    sft_data = tmp_path / "sft.jsonl"
    conv = [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好，有什么可以帮你？"}]
    sft_data.write_text("\n".join(json.dumps({"conversations": conv}, ensure_ascii=False) for _ in range(8)), encoding="utf-8")
    out = _run(
        "algo.train_sft",
        ["--arch", "miniseek-v", "--data", str(sft_data), "--init-ckpt", str(pt_out / "last.pt"), "--out-dir",
         str(tmp_path / "sft"), "--max-steps", "2", "--batch-size", "2", "--max-seq-len", "32", "--no-compile",
         "--log-every", "1"],
        env,
    )
    assert "candidate_pool=True bounded_replay=True" in out
    assert (tmp_path / "sft" / "last.pt").exists()
