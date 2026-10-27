"""采样策略：温度 / top-k / top-p / 重复惩罚 / 禁复 n-gram。

欠训的小模型最常见的失败是"复读机"——纯 top-k 采样几乎必然掉进循环。
这几项约束集中在这里实现，dense 与 moe 共用，避免两份逻辑各自漂移。
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F

DEFAULT_REP_WINDOW = 128


def _penalize_repeats(logits: torch.Tensor, history: Sequence[int], penalty: float, window: int) -> None:
    """对最近 window 个 token 的 logits 做除法/乘法惩罚（原地）。"""
    if penalty is None or abs(float(penalty) - 1.0) < 1e-6 or not len(history):
        return
    recent = history[-max(1, int(window)):] if window else history
    ids = sorted(set(int(x) for x in recent))
    idx = torch.tensor(ids, device=logits.device, dtype=torch.long)
    vals = logits.index_select(-1, idx)
    vals = torch.where(vals > 0, vals / penalty, vals * penalty)
    logits.index_copy_(-1, idx, vals)


def _ban_ngram_repeats(logits: torch.Tensor, history: Sequence[int], n: int) -> None:
    """禁止重复出现过的 n-gram：把 (当前前缀 -> 后续 token) 的候选位置置 -inf。"""
    if not n or int(n) <= 1 or len(history) < int(n):
        return
    n = int(n)
    prefix = tuple(int(x) for x in history[-(n - 1):])
    banned = set()
    for i in range(len(history) - n + 1):
        if tuple(int(x) for x in history[i:i + n - 1]) == prefix:
            banned.add(int(history[i + n - 1]))
    if banned:
        idx = torch.tensor(sorted(banned), device=logits.device, dtype=torch.long)
        logits.index_fill_(0, idx, float("-inf"))


def _top_k(logits: torch.Tensor, k: Optional[int]) -> torch.Tensor:
    if not k or int(k) <= 0:
        return logits
    kk = min(int(k), logits.numel())
    v, _ = torch.topk(logits, kk)
    return logits.masked_fill(logits < v[-1], float("-inf"))


def _top_p(logits: torch.Tensor, p: Optional[float]) -> torch.Tensor:
    if p is None or not (0.0 < float(p) < 1.0):
        return logits
    s, _ = torch.sort(logits, descending=True)
    cum = torch.cumsum(F.softmax(s, dim=-1), dim=-1)
    over = (cum > float(p)).nonzero()
    if len(over) == 0:
        return logits
    thr = s[int(over[0])]
    return logits.masked_fill(logits < thr, float("-inf"))


def sample_next(
    logits: torch.Tensor,
    history: Sequence[int],
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    repetition_window: int = DEFAULT_REP_WINDOW,
    no_repeat_ngram_size: int = 0,
) -> torch.Tensor:
    """从最后一步 logits 采样一个 token，返回形状 [1, 1]。

    logits: 形如 [V] 的一维张量（不会被原地修改）。
    history: 目前已有的完整 token 序列（prompt + 已生成）。
    """
    raw = logits
    lg = logits.clone() / max(float(temperature), 1e-6)
    _penalize_repeats(lg, history, float(repetition_penalty or 1.0), int(repetition_window or 0))
    _ban_ngram_repeats(lg, history, int(no_repeat_ngram_size or 0))
    lg = _top_k(lg, top_k)
    lg = _top_p(lg, top_p)
    if not bool(torch.isfinite(lg).any()):
        # 全被屏蔽（极端情况）时退回原始分布，至少不崩
        lg = raw.clone() / max(float(temperature), 1e-6)
    probs = F.softmax(lg, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(1, 1)
