"""Length-bucket and document-packing batch helpers for pretrain speed.

Bench measured (effective tok/s vs random pad):
  bucket ~+30% (pad ~1%), pack ~+40% (pad ~0%).
Packing joins full docs with eos (no per-doc pre-truncate) and cuts the stream
into non-overlapping max_seq_len chunks. By default there is no cross-document
attention mask (causal attention sees across doc boundaries). With
``with_doc_ids=True`` (v41) docs are padded to ``doc_align`` and every batch
carries per-token doc ids for sample-level attention masking; predictions that
cross a doc boundary are dropped from the loss.
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


def append_aligned_doc(
    stream_ids: List[int],
    stream_doc: List[int],
    stream_lab: List[int],
    ids: Sequence[int],
    labs: Sequence[int],
    doc_id: int,
    align: int,
    pad_id: int,
) -> None:
    """追加一篇文档（含其 eos）；长度补齐到 align 的倍数，补的 pad 不算损失、归属本文档。"""
    stream_ids.extend(int(t) for t in ids)
    stream_lab.extend(int(t) for t in labs)
    stream_doc.extend([int(doc_id)] * len(ids))
    rem = len(ids) % max(1, int(align))
    if rem:
        n = int(align) - rem
        stream_ids.extend([int(pad_id)] * n)
        stream_lab.extend([-100] * n)
        stream_doc.extend([int(doc_id)] * n)


def stack_doc_pack_group(
    group: Sequence[Tuple[Sequence[int], Sequence[int], Sequence[int]]],
    max_seq: int,
    pad_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(ids, doc, lab) 切块 → x [B,S-1]、y [B,S-1]、doc_ids [B,S-1]。

    y[t] = lab[t+1]，若 t+1 属于另一篇文档则为 −100（样本级 attention mask 下不跨文档预测）。
    doc_ids 在块内从 0 重新编号；尾部补齐的 pad 自成一篇。
    """
    if not group:
        raise ValueError("empty pack group")
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    ds: List[torch.Tensor] = []
    for ids, docs, labs in group:
        ids, docs, labs = [int(t) for t in ids], [int(t) for t in docs], [int(t) for t in labs]
        if len(ids) < max_seq:
            n = max_seq - len(ids)
            nxt = (docs[-1] + 1) if docs else 0
            ids += [int(pad_id)] * n
            labs += [-100] * n
            docs += [nxt] * n
        ids, docs, labs = ids[:max_seq], docs[:max_seq], labs[:max_seq]
        base = docs[0]
        docs = [d - base for d in docs]
        y = [labs[t + 1] if docs[t + 1] == docs[t] else -100 for t in range(max_seq - 1)]
        xs.append(torch.tensor(ids[:-1], dtype=torch.long))
        ys.append(torch.tensor(y, dtype=torch.long))
        ds.append(torch.tensor(docs[:-1], dtype=torch.long))
    return torch.stack(xs), torch.stack(ys), torch.stack(ds)


def _iter_doc_packs(encoded, *, max_seq: int, batch_size: int, align: int, pad_id: int):
    """encoded 产出 (ids, labs)；对齐打包后按 batch 产出 (x, y, doc_ids)。"""
    if max_seq % max(1, int(align)):
        raise ValueError(f"max_seq_len {max_seq} 必须是 doc_align {align} 的倍数")
    s_ids: List[int] = []
    s_doc: List[int] = []
    s_lab: List[int] = []
    ready: List[Tuple[List[int], List[int], List[int]]] = []
    for doc_id, (ids, labs) in enumerate(encoded):
        append_aligned_doc(s_ids, s_doc, s_lab, ids, labs, doc_id, align, pad_id)
        while len(s_ids) >= max_seq:
            ready.append((s_ids[:max_seq], s_doc[:max_seq], s_lab[:max_seq]))
            s_ids, s_doc, s_lab = s_ids[max_seq:], s_doc[max_seq:], s_lab[max_seq:]
            if len(ready) >= batch_size:
                yield stack_doc_pack_group(ready[:batch_size], max_seq=max_seq, pad_id=pad_id)
                ready = ready[batch_size:]
    if len(s_ids) >= 2:
        ready.append((s_ids, s_doc, s_lab))
    for i in range(0, len(ready) - batch_size + 1, batch_size):
        yield stack_doc_pack_group(ready[i : i + batch_size], max_seq=max_seq, pad_id=pad_id)


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

    Encodes on the fly (full docs, no per-doc truncate), appends eos between
    docs, cuts the stream into non-overlapping max_seq_len chunks, emits [B, S]
    batches. Model context is still max_seq_len; long tails appear in later chunks.
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
        doc_align: int = 1,
        with_doc_ids: bool = False,
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
        # M：文档对齐到压缩比的倍数，并产出逐 token 文档编号（样本级 attention mask）
        self.doc_align = int(doc_align)
        self.with_doc_ids = bool(with_doc_ids)

    def _encoded_docs(self, order: Sequence[int]):
        for idx in order:
            ids = self.tokenizer.encode(self.texts[idx], add_special_tokens=True)
            if not ids:
                continue
            ids = list(ids) + [self.eos_id]
            yield ids, ids

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        rng = random.Random(self.seed)
        order = list(range(len(self.texts)))
        rng.shuffle(order)
        if self.with_doc_ids:
            yield from _iter_doc_packs(
                self._encoded_docs(order),
                max_seq=self.max_seq_len,
                batch_size=self.batch_size,
                align=self.doc_align,
                pad_id=self.pad_id,
            )
            return
        stream: List[int] = []
        ready: List[List[int]] = []
        for idx in order:
            text = self.texts[idx]
            ids = self.tokenizer.encode(text, add_special_tokens=True)
            if not ids:
                continue
            # Keep the full document; the pack cutter below emits non-overlapping
            # max_seq_len chunks so long-doc tails land in later windows instead
            # of being dropped. (random/bucket still truncate per sample.)
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
        # char≈token for this Chinese BPE; count full docs (pack no longer
        # drops tails), so LR/horizon matches real token volume.
        avg = sum(max(2, len(t)) for t in self.texts) / len(self.texts)
        return estimate_pack_steps(
            len(self.texts), avg, self.batch_size, self.max_seq_len
        )

def truncate_keep_tail_with_eos(
    ids: Sequence[int],
    max_seq_len: int,
    eos_id: int,
    labels: Optional[Sequence[int]] = None,
) -> Tuple[List[int], Optional[List[int]]]:
    """Keep the tail of a token sequence; preserve a trailing eos when present.

    If the untruncated sequence ended with ``eos_id``, the returned sequence also
    ends with ``eos_id`` (forced onto the last position if needed). Labels, when
    provided, are truncated in lockstep; a forced trailing eos uses label=eos_id
    so SFT still trains the stop token.
    """
    if max_seq_len < 2:
        raise ValueError("max_seq_len must be >= 2")
    ids_list = [int(t) for t in ids]
    lab_list: Optional[List[int]] = None
    if labels is not None:
        lab_list = [int(t) for t in labels]
        if len(lab_list) != len(ids_list):
            raise ValueError("ids/labels length mismatch")
    ended_with_eos = bool(ids_list) and ids_list[-1] == int(eos_id)
    if len(ids_list) > max_seq_len:
        ids_list = ids_list[-max_seq_len:]
        if lab_list is not None:
            lab_list = lab_list[-max_seq_len:]
    if ended_with_eos:
        if not ids_list:
            ids_list = [int(eos_id)]
            if lab_list is not None:
                lab_list = [int(eos_id)]
        else:
            ids_list[-1] = int(eos_id)
            if lab_list is not None:
                # Keep stop-token in the loss when the sample ended with eos.
                lab_list[-1] = int(eos_id)
    return ids_list, lab_list




def truncate_keep_head_with_eos(
    ids: Sequence[int],
    max_seq_len: int,
    eos_id: int,
    labels: Optional[Sequence[int]] = None,
) -> Tuple[List[int], Optional[List[int]]]:
    """Keep the head of a token sequence; drop overflow from the end.

    Prefer preserving the prompt/question at the start over the completion tail.
    Always force a trailing ``eos_id`` on the kept window so SFT still trains the
    stop token (needed when truncation cuts off the real ending eos). Labels
    truncate in lockstep; forced trailing eos uses label=eos_id.
    """
    if max_seq_len < 2:
        raise ValueError("max_seq_len must be >= 2")
    ids_list = [int(t) for t in ids]
    lab_list: Optional[List[int]] = None
    if labels is not None:
        lab_list = [int(t) for t in labels]
        if len(lab_list) != len(ids_list):
            raise ValueError("ids/labels length mismatch")
    if len(ids_list) > max_seq_len:
        ids_list = ids_list[:max_seq_len]
        if lab_list is not None:
            lab_list = lab_list[:max_seq_len]
    if not ids_list:
        ids_list = [int(eos_id)]
        if lab_list is not None:
            lab_list = [int(eos_id)]
    else:
        ids_list[-1] = int(eos_id)
        if lab_list is not None:
            lab_list[-1] = int(eos_id)
    return ids_list, lab_list


def stack_sft_pack_group(
    group: Sequence[Tuple[Sequence[int], Sequence[int]]],
    max_seq: int,
    pad_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack packed SFT (ids, labels) chunks into [B, max_seq-1] x/y.

    Pad ids with pad_id and labels with -100 beyond real_len; then
    x=ids[:-1], y=labels[1:] (pad positions in y already -100).
    """
    if not group:
        raise ValueError("empty SFT pack group")
    pad = int(pad_id)
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    for ids, labs in group:
        id_list = [int(t) for t in ids]
        lab_list = [int(t) for t in labs]
        if len(id_list) != len(lab_list):
            raise ValueError("ids/labels length mismatch in pack chunk")
        real_len = min(len(id_list), max_seq)
        if len(id_list) < max_seq:
            pad_n = max_seq - len(id_list)
            id_list = id_list + [pad] * pad_n
            lab_list = lab_list + [-100] * pad_n
        else:
            id_list = id_list[:max_seq]
            lab_list = lab_list[:max_seq]
        x_ids = id_list[:-1]
        y_labs = []
        for j, tok in enumerate(lab_list[1:]):
            if j + 1 >= real_len:
                y_labs.append(-100)
            else:
                y_labs.append(int(tok))
        xs.append(torch.tensor(x_ids, dtype=torch.long))
        ys.append(torch.tensor(y_labs, dtype=torch.long))
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)


class PackedSFTBatchIter:
    """Online packer for SFT (ids, labels) sequences (one epoch).

    Encodes via ``encode_fn(sample) -> (ids, labels)`` (full sequences, already
    truncated). Concatenates samples into a stream, cuts max_seq windows, stacks
    batch_size chunks. No cross-sample attention mask (same as pretrain pack).
    """

    def __init__(
        self,
        samples: Sequence,
        encode_fn,
        *,
        max_seq_len: int,
        batch_size: int,
        pad_id: int = 0,
        seed: int = 0,
        doc_align: int = 1,
        with_doc_ids: bool = False,
    ) -> None:
        self.samples = list(samples)
        self.encode_fn = encode_fn
        self.max_seq_len = int(max_seq_len)
        self.batch_size = int(batch_size)
        self.pad_id = int(pad_id)
        self.seed = int(seed)
        self.doc_align = int(doc_align)
        self.with_doc_ids = bool(with_doc_ids)

    def _encoded_samples(self, order: Sequence[int]):
        for idx in order:
            ids, labs = self.encode_fn(self.samples[idx])
            if not ids or len(ids) < 2:
                continue
            if len(ids) != len(labs):
                raise ValueError("encode_fn returned mismatched ids/labels")
            yield list(ids), list(labs)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        rng = random.Random(self.seed)
        order = list(range(len(self.samples)))
        rng.shuffle(order)
        if self.with_doc_ids:
            yield from _iter_doc_packs(
                self._encoded_samples(order),
                max_seq=self.max_seq_len,
                batch_size=self.batch_size,
                align=self.doc_align,
                pad_id=self.pad_id,
            )
            return
        stream_ids: List[int] = []
        stream_labs: List[int] = []
        ready: List[Tuple[List[int], List[int]]] = []
        for idx in order:
            ids, labs = self.encode_fn(self.samples[idx])
            if not ids or len(ids) < 2:
                continue
            if len(ids) != len(labs):
                raise ValueError("encode_fn returned mismatched ids/labels")
            stream_ids.extend(int(t) for t in ids)
            stream_labs.extend(int(t) for t in labs)
            while len(stream_ids) >= self.max_seq_len:
                chunk_i = stream_ids[: self.max_seq_len]
                chunk_l = stream_labs[: self.max_seq_len]
                stream_ids = stream_ids[self.max_seq_len :]
                stream_labs = stream_labs[self.max_seq_len :]
                ready.append((chunk_i, chunk_l))
                if len(ready) >= self.batch_size:
                    group = ready[: self.batch_size]
                    ready = ready[self.batch_size :]
                    yield stack_sft_pack_group(
                        group, max_seq=self.max_seq_len, pad_id=self.pad_id
                    )
        if len(stream_ids) >= 2:
            ready.append((stream_ids, stream_labs))
        for i in range(0, len(ready) - self.batch_size + 1, self.batch_size):
            yield stack_sft_pack_group(
                ready[i : i + self.batch_size],
                max_seq=self.max_seq_len,
                pad_id=self.pad_id,
            )

    def estimate_len(self) -> int:
        if not self.samples:
            return 1
        # Cheap char proxy (same idea as PackedTextBatchIter)
        avg = 0.0
        n = 0
        for s in self.samples[: min(256, len(self.samples))]:
            try:
                ids, _labs = self.encode_fn(s)
            except Exception:
                continue
            if ids:
                avg += float(len(ids))
                n += 1
        if n <= 0:
            avg = float(self.max_seq_len)
        else:
            avg /= float(n)
        return estimate_pack_steps(
            len(self.samples), avg, self.batch_size, self.max_seq_len
        )
