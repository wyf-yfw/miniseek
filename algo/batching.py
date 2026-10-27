"""Length-bucket and document-packing batch helpers for pretrain speed.

Bench measured (effective tok/s vs random pad):
  bucket ~+30% (pad ~1%), pack ~+40% (pad ~0%).
Packing joins docs with eos and does NOT add cross-document attention masks
(same as the speed bench). Causal attention still sees across doc boundaries.
"""

from __future__ import annotations

import random
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Sampler


def sort_indices_by_length(lengths: Sequence[int]) -> List[int]:
    return sorted(range(len(lengths)), key=lambda i: int(lengths[i]))


def make_length_batches(
    lengths: Sequence[int],
    batch_size: int,
    *,
    seed: int = 0,
    shuffle_batches: bool = True,
    drop_last: bool = False,
) -> List[List[int]]:
    """Group indices sorted by length into batches; optionally shuffle batch order."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    order = sort_indices_by_length(lengths)
    batches: List[List[int]] = []
    for i in range(0, len(order), batch_size):
        chunk = list(order[i : i + batch_size])
        if not chunk:
            continue
        if drop_last and len(chunk) < batch_size:
            continue
        batches.append(chunk)
    if shuffle_batches and batches:
        rng = random.Random(int(seed))
        rng.shuffle(batches)
    return batches


class LengthBucketBatchSampler(Sampler[List[int]]):
    """Yield index lists of similar length; reshuffle batch order each epoch."""

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        *,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        self.lengths = [int(x) for x in lengths]
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def __iter__(self) -> Iterator[List[int]]:
        batches = make_length_batches(
            self.lengths,
            self.batch_size,
            seed=self.seed + self.epoch,
            shuffle_batches=True,
            drop_last=self.drop_last,
        )
        self.epoch += 1
        yield from batches

    def __len__(self) -> int:
        n = len(self.lengths)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


def pack_docs_to_stream(docs: Iterable[Sequence[int]], eos_id: int) -> List[int]:
    stream: List[int] = []
    eos = int(eos_id)
    for ids in docs:
        if not ids:
            continue
        stream.extend(int(t) for t in ids)
        stream.append(eos)
    return stream


def cut_packed_stream(stream: Sequence[int], max_seq: int) -> List[List[int]]:
    """Cut a token stream into chunks of length <= max_seq (drop trailing <2)."""
    if max_seq < 2:
        raise ValueError("max_seq must be >= 2")
    chunks: List[List[int]] = []
    for i in range(0, max(0, len(stream) - 1), max_seq):
        chunk = list(stream[i : i + max_seq])
        if len(chunk) < 2:
            continue
        chunks.append(chunk)
    return chunks


def stack_pack_group(
    group: Sequence[Sequence[int]],
    max_seq: int,
    pad_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack packed chunks into [B, max_seq-1] x/y; pad y with -100 beyond real len.

    Each chunk is padded/truncated to max_seq, then x=ids[:-1], y=ids[1:] with
    padded positions in y set to -100.
    """
    if not group:
        raise ValueError("empty pack group")
    pad = int(pad_id)
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    for chunk in group:
        ids = list(chunk)
        real_len = min(len(ids), max_seq)
        if len(ids) < max_seq:
            ids = ids + [pad] * (max_seq - len(ids))
        else:
            ids = ids[:max_seq]
        x_ids = ids[:-1]
        y_ids = ids[1:]
        y_t = []
        for j, tok in enumerate(y_ids):
            if j + 1 >= real_len:
                y_t.append(-100)
            else:
                y_t.append(int(tok))
        xs.append(torch.tensor(x_ids, dtype=torch.long))
        ys.append(torch.tensor(y_t, dtype=torch.long))
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)


def build_pack_batches(
    enc: Sequence[Sequence[int]],
    batch_size: int,
    max_seq: int,
    *,
    eos_id: int = 2,
    pad_id: int = 0,
    seed: int = 0,
    shuffle_docs: bool = True,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """eos between docs → cut max_seq → stack batch_size sequences [B, S]."""
    docs = list(enc)
    if shuffle_docs:
        rng = random.Random(int(seed))
        rng.shuffle(docs)
    stream = pack_docs_to_stream(docs, eos_id=eos_id)
    chunks = cut_packed_stream(stream, max_seq=max_seq)
    batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for i in range(0, len(chunks) - batch_size + 1, batch_size):
        group = chunks[i : i + batch_size]
        batches.append(stack_pack_group(group, max_seq=max_seq, pad_id=pad_id))
    return batches


def estimate_pack_steps(
    n_docs: int,
    avg_tokens_per_doc: float,
    batch_size: int,
    max_seq: int,
) -> int:
    """Rough step count for LR schedule when using packing."""
    total_tok = max(0.0, float(n_docs) * float(avg_tokens_per_doc))
    # each doc adds one eos
    total_tok += float(n_docs)
    per_batch = float(batch_size) * float(max(1, max_seq))
    if per_batch <= 0:
        return 1
    return max(1, int(total_tok // per_batch))


class PackedTextBatchIter:
    """Online packer over a list of raw texts (one epoch).

    Encodes on the fly, appends eos between docs, emits [B, S] batches.
    """

    def __init__(
        self,
        texts: Sequence[str],
        tokenizer,
        *,
        max_seq_len: int,
        batch_size: int,
        pad_id: int = 0,
        eos_id: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)
        self.batch_size = int(batch_size)
        self.pad_id = int(pad_id)
        self.eos_id = int(
            eos_id
            if eos_id is not None
            else (getattr(tokenizer, "eos_token_id", None) or 2)
        )
        self.seed = int(seed)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        rng = random.Random(self.seed)
        order = list(range(len(self.texts)))
        rng.shuffle(order)
        stream: List[int] = []
        ready: List[List[int]] = []
        for idx in order:
            text = self.texts[idx]
            ids = self.tokenizer.encode(text, add_special_tokens=True)
            if not ids:
                continue
            # Match JsonlTextDataset truncation (keep trailing eos when present)
            if len(ids) > self.max_seq_len:
                if ids[-1] == self.eos_id:
                    ids = ids[: self.max_seq_len - 1] + [self.eos_id]
                else:
                    ids = ids[: self.max_seq_len]
            stream.extend(ids)
            stream.append(self.eos_id)
            while len(stream) >= self.max_seq_len:
                chunk = stream[: self.max_seq_len]
                stream = stream[self.max_seq_len :]
                ready.append(chunk)
                if len(ready) >= self.batch_size:
                    group = ready[: self.batch_size]
                    ready = ready[self.batch_size :]
                    yield stack_pack_group(
                        group, max_seq=self.max_seq_len, pad_id=self.pad_id
                    )
        if len(stream) >= 2:
            ready.append(stream)
        # Drop incomplete final batch (same as bench) to keep shapes stable
        for i in range(0, len(ready) - self.batch_size + 1, self.batch_size):
            yield stack_pack_group(
                ready[i : i + self.batch_size],
                max_seq=self.max_seq_len,
                pad_id=self.pad_id,
            )

    def estimate_len(self) -> int:
        if not self.texts:
            return 1
        # char≈token for this Chinese BPE; good enough for LR horizon
        avg = sum(min(self.max_seq_len, max(2, len(t))) for t in self.texts) / len(
            self.texts
        )
        return estimate_pack_steps(
            len(self.texts), avg, self.batch_size, self.max_seq_len
        )
