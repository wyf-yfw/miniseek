"""MiniMind tokenizer 加载。

实测：jingyaogong 自定义 minimind_tokenizer 词表大小为 **6400**
（非旧版 mistral_tokenizer 的 ~32000）。本仓库默认与之对齐。

下载方式（任选其一）::

    # Hugging Face
    huggingface-cli download jingyaogong/MiniMind2 --include "tokenizer*" --local-dir ./tokenizer

    # 或从 MiniMind 仓库拷贝 minimind_tokenizer 目录
    # https://github.com/jingyaogong/minimind

环境变量 ``DS41_TOKENIZER_PATH`` 可覆盖本地路径。
无 transformers / 无本地文件时，提供 Byte-level 占位 tokenizer 供 CPU smoke。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Union

# MiniMind 官方自定义词表（2024-09 起统一）
MINIMIND_VOCAB_SIZE = 6400
DEFAULT_HF_TOKENIZER = "jingyaogong/MiniMind2"
DEFAULT_LOCAL_CANDIDATES = (
    "./tokenizer",
    "./minimind_tokenizer",
    os.path.expanduser("~/.cache/ds41-mini/tokenizer"),
)


class FallbackTokenizer:
    """极简占位：按 UTF-8 字节映射到 [3, 258)，vocab_size=6400 兼容 embedding.

    仅用于无网络/无 HF 时的 import 与 CPU smoke，不可用于正式训练。
    """

    def __init__(self, vocab_size: int = MINIMIND_VOCAB_SIZE) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
        self.name_or_path = "fallback-byte-tokenizer"

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        ids = [b + 3 for b in text.encode("utf-8")]
        ids = [i if i < self.vocab_size else self.unk_token_id for i in ids]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        special = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        raw = []
        for i in ids:
            i = int(i)
            if skip_special_tokens and i in special:
                continue
            if 3 <= i < 259:
                raw.append(i - 3)
        try:
            return bytes(raw).decode("utf-8", errors="replace")
        except Exception:
            return ""

    def __call__(
        self,
        text: Union[str, List[str]],
        return_tensors: Optional[str] = None,
        add_special_tokens: bool = True,
        **kwargs,
    ):
        import torch

        if isinstance(text, str):
            ids = [self.encode(text, add_special_tokens=add_special_tokens)]
        else:
            ids = [self.encode(t, add_special_tokens=add_special_tokens) for t in text]
        if return_tensors == "pt":
            max_len = max(len(x) for x in ids)
            batch = torch.full((len(ids), max_len), self.pad_token_id, dtype=torch.long)
            for i, row in enumerate(ids):
                batch[i, : len(row)] = torch.tensor(row, dtype=torch.long)
            return {"input_ids": batch}
        return {"input_ids": ids}


def _try_load_hf(path_or_id: str):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    try:
        return AutoTokenizer.from_pretrained(path_or_id, trust_remote_code=True)
    except Exception:
        return None


def load_tokenizer(
    path: Optional[str] = None,
    *,
    allow_fallback: bool = True,
    prefer_online: bool = False,
):
    """加载 MiniMind tokenizer；失败时可回退到 FallbackTokenizer.

    返回对象至少具备：``vocab_size``, ``encode``, ``decode``, ``eos_token_id``.
    """
    env = os.environ.get("DS41_TOKENIZER_PATH")
    candidates: List[str] = []
    if path:
        candidates.append(path)
    if env:
        candidates.append(env)
    for c in DEFAULT_LOCAL_CANDIDATES:
        if Path(c).exists():
            candidates.append(c)
    if prefer_online:
        candidates.append(DEFAULT_HF_TOKENIZER)

    for c in candidates:
        tok = _try_load_hf(c)
        if tok is not None:
            vs = int(getattr(tok, "vocab_size", None) or len(tok))
            print(f"[tokenizer] loaded={c!r} vocab_size={vs}")
            if vs != MINIMIND_VOCAB_SIZE:
                print(
                    f"[tokenizer] WARN: expected MiniMind vocab={MINIMIND_VOCAB_SIZE}, "
                    f"got {vs} — ModelConfig.vocab_size 应与此一致"
                )
            return tok

    if not prefer_online:
        tok = _try_load_hf(DEFAULT_HF_TOKENIZER)
        if tok is not None:
            vs = int(getattr(tok, "vocab_size", None) or len(tok))
            print(f"[tokenizer] loaded={DEFAULT_HF_TOKENIZER!r} vocab_size={vs}")
            return tok

    if allow_fallback:
        print(
            "[tokenizer] MiniMind tokenizer 未找到，使用 FallbackTokenizer "
            f"(vocab={MINIMIND_VOCAB_SIZE})；正式训练请下载 jingyaogong/MiniMind2 tokenizer"
        )
        return FallbackTokenizer(MINIMIND_VOCAB_SIZE)

    raise FileNotFoundError(
        "无法加载 MiniMind tokenizer。请设置 DS41_TOKENIZER_PATH 或下载到 ./tokenizer\n"
        f"  huggingface-cli download {DEFAULT_HF_TOKENIZER} --include 'tokenizer*' --local-dir ./tokenizer"
    )


def detect_vocab_size(tokenizer=None) -> int:
    if tokenizer is None:
        tokenizer = load_tokenizer(allow_fallback=True)
    return int(getattr(tokenizer, "vocab_size", None) or len(tokenizer))
