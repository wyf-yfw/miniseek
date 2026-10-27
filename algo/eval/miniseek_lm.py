"""MiniSeek custom model for lm-evaluation-harness."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from tqdm import tqdm

from algo.model.config import ModelConfig
from algo.model.dense import DenseLM
from algo.model.tokenizer import load_tokenizer

try:
    from lm_eval.api.model import TemplateLM
    from lm_eval.api.registry import register_model
except ModuleNotFoundError:  # pragma: no cover - optional dep for harness
    TemplateLM = object  # type: ignore[misc,assignment]

    def register_model(*_a, **_k):  # type: ignore[misc]
        def deco(cls):
            return cls
        return deco



def _update_sample_progress(done: int, total: int) -> None:
    """Best-effort sample progress for the Web UI (eval/runs/<id>/progress.json)."""
    import os
    import json
    p = os.environ.get("BENCH_PROGRESS_PATH")
    if not p:
        return
    path = Path(p)
    try:
        data = {}
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
        data["sample_done"] = int(done)
        data["sample_total"] = int(total)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def load_miniseek_blob(path: Union[str, Path]):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir() and (path / "pytorch_model.bin").exists():
        weight_path = path / "pytorch_model.bin"
        cfg_path = path / "config.json"
    elif path.name == "pytorch_model.bin":
        weight_path = path
        cfg_path = path.parent / "config.json"
    else:
        weight_path = None
        cfg_path = None
    if weight_path is not None:
        sd = torch.load(weight_path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model" in sd:
            blob = sd
        else:
            cfg_raw = (
                json.loads(cfg_path.read_text(encoding="utf-8"))
                if cfg_path and cfg_path.exists()
                else {}
            )
            allowed = set(ModelConfig.__annotations__)
            cfg_dict = {k: v for k, v in cfg_raw.items() if k in allowed}
            blob = {"model": sd, "config": cfg_dict}
    else:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(blob["config"])
    if getattr(cfg, "arch", "dense") == "moe" or getattr(cfg, "use_moe", False):
        from algo.model.moe import MoELM

        model = MoELM(cfg)
    else:
        model = DenseLM(cfg)
    model.load_state_dict(blob["model"], strict=True)
    model.eval()
    tok = load_tokenizer(allow_fallback=True)
    return model, tok, cfg


@register_model("miniseek")
class MiniSeekLM(TemplateLM):
    """Causal MiniSeek Dense/MoE wrapper for lm-eval."""

    def __init__(
        self,
        ckpt: str,
        device: str = "cpu",
        batch_size: int = 1,
        max_length: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._device = torch.device(device)
        self.model, self.tokenizer, self.cfg = load_miniseek_blob(ckpt)
        self.model.to(self._device)
        self.model.eval()
        self._batch_size = int(batch_size)
        self._max_length = int(max_length or getattr(self.cfg, "max_seq_len", 512) or 512)
        self.ckpt = ckpt

    @classmethod
    def create_from_arg_string(cls, arg_string: str, additional_config: Optional[dict] = None):
        args: dict[str, Any] = {}
        if arg_string:
            for part in arg_string.split(","):
                part = part.strip()
                if not part:
                    continue
                if "=" not in part:
                    raise ValueError(f"bad model_args piece: {part}")
                k, v = part.split("=", 1)
                args[k.strip()] = v.strip()
        if additional_config:
            args.update({k: v for k, v in additional_config.items() if v is not None})
        if "batch_size" in args:
            args["batch_size"] = int(args["batch_size"])
        if "max_length" in args:
            args["max_length"] = int(args["max_length"])
        return cls(**args)

    @property
    def eot_token_id(self) -> int:
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if eos is None:
            return 2
        return int(eos)

    @property
    def device(self):
        return self._device

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def rank(self) -> int:
        return 0

    @property
    def world_size(self) -> int:
        return 1

    def tok_encode(
        self, string: str, add_special_tokens: bool | None = None, **kwargs
    ) -> list[int]:
        if add_special_tokens is None:
            add_special_tokens = False
        return list(
            self.tokenizer.encode(string, add_special_tokens=bool(add_special_tokens))
        )

    def tok_decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str], list[int], list[int]]],
        disable_tqdm: bool = False,
        **kwargs: Any,
    ) -> list[tuple[float, bool]]:
        results: list[tuple[float, bool]] = []
        total = len(requests)
        _update_sample_progress(0, total)
        for i, (_key, context_enc, continuation_enc) in enumerate(
            tqdm(requests, disable=disable_tqdm, desc="loglikelihood")
        ):
            if not continuation_enc:
                results.append((0.0, True))
                _update_sample_progress(i + 1, total)
                continue
            inp = (context_enc + continuation_enc)[-(self.max_length + 1) :]
            # Leave at least one continuation token for scoring
            cont_len = len(continuation_enc)
            if len(inp) < cont_len + 1:
                # truncate continuation from the left if needed
                cont_len = max(1, len(inp) - 1)
            input_ids = torch.tensor([inp], dtype=torch.long, device=self._device)
            with torch.no_grad():
                logits, _ = self.model(input_ids)
            # logits at position i predict token i+1
            # score continuation tokens: last cont_len tokens of inp
            logits = logits[0, -cont_len - 1 : -1, :]
            target = input_ids[0, -cont_len:]
            log_probs = F.log_softmax(logits.float(), dim=-1)
            gathered = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
            ll = float(gathered.sum().item())
            greedy = logits.argmax(dim=-1)
            is_greedy = bool(torch.equal(greedy, target))
            results.append((ll, is_greedy))
            _update_sample_progress(i + 1, total)
        return results

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("loglikelihood_rolling not used for MiniSeek bench suite")

    def generate_until(self, requests, disable_tqdm: bool = False):
        res: list[str] = []
        total = len(requests)
        _update_sample_progress(0, total)
        for i, req in enumerate(tqdm(requests, disable=disable_tqdm, desc="generate_until")):
            context, gen_kwargs = req.args
            until = gen_kwargs.get("until") or []
            if isinstance(until, str):
                until = [until]
            max_gen = int(gen_kwargs.get("max_gen_toks", 64))
            ctx_ids = self.tok_encode(context, add_special_tokens=True)
            # Keep room for at least one new token on the first step
            ctx_ids = ctx_ids[-(self.max_length - 1) :]
            ids = list(ctx_ids)
            eos = self.eot_token_id
            for _ in range(max_gen):
                # Sliding window: never feed more than max_seq_len (ids can grow)
                window = ids[-self.max_length :]
                input_ids = torch.tensor([window], dtype=torch.long, device=self._device)
                with torch.no_grad():
                    logits, _ = self.model(input_ids)
                next_id = int(logits[0, -1, :].argmax().item())
                ids.append(next_id)
                text = self.tok_decode(ids[len(ctx_ids) :])
                stop = False
                for u in until:
                    if u and u in text:
                        text = text.split(u)[0]
                        stop = True
                        break
                if stop or next_id == eos:
                    res.append(text)
                    break
            else:
                res.append(self.tok_decode(ids[len(ctx_ids) :]))
            _update_sample_progress(i + 1, total)
        return res
