"""Teach GLM-5 Next's ``load_weights`` the plain EXL3 tensor names.

Routing by field suffix after the ``model.language_model.`` strip:
  * routed experts (layers 3--44)  -> ``Exl3MoEMethod`` payload parameters;
  * native dense modules           -> ``Exl3LinearMethod`` payload parameters;
  * all other dense trellis        -> decoded to BF16 ``<module>.weight`` and handed to
                                      the stock loader (sanctioned fallback, dense only);
  * NEXTN layer 45 tensors         -> dropped while MTP is off, counted;
  * vision q/k/v trellis           -> dropped in favour of the BF16 fused ``attn.qkv``, counted;
  * everything else                -> stock loader unchanged.
Any trellis tensor that reaches the end unrouted aborts the load.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Iterable

import torch

from sglang.srt.runtime_context import get_parallel

from .contract import EXPERTS, FIRST_MOE_LAYER, LAST_MOE_LAYER, NEXTN_LAYER, is_vision_qkv_duplicate, parse, strip_prefix
from .dequant import ByteAccount, dequantize, resolve_decoder

log = logging.getLogger("exl3_plain")


def wrap_load_weights(original):
    def wrapped(self, weights: Iterable[tuple[str, torch.Tensor]], is_nextn: bool = False):
        if is_nextn:
            raise NotImplementedError("NEXTN/MTP on exl3_plain is a follow-up experiment; launch with MTP off (see LAUNCH.md)")
        if get_parallel().tp_size != 1:
            raise ValueError("exl3_plain loads TP=1 only")
        config = _quant_config(self)
        codebook = config.census["codebook"]
        params = dict(self.named_parameters())
        acct, pending, decoder, logged = ByteAccount(), defaultdict(dict), None, set()
        remaining: list[tuple[str, torch.Tensor]] = []

        for source_name, tensor in weights:
            key = parse(source_name)
            if key is None:
                if is_vision_qkv_duplicate(source_name):
                    acct.add("dropped:vision_qkv_duplicate", tensor, 0)
                    continue
                if source_name.startswith(f"model.language_model.layers.{NEXTN_LAYER}."):
                    acct.add("dropped:nextn_mtp_off", tensor, 0)
                    continue
                acct.add("bf16:passthrough", tensor)
                remaining.append((source_name, tensor))
                continue
            if key.layer == NEXTN_LAYER:
                acct.add("dropped:nextn_mtp_off", tensor, 0)
                continue
            if key.kind == "expert":
                if not FIRST_MOE_LAYER <= key.layer <= LAST_MOE_LAYER or not 0 <= key.expert < EXPERTS:
                    raise ValueError(f"expert tensor outside the routed pool: {source_name}")
                group = "w13" if key.projection in ("gate_proj", "up_proj") else "w2"
                target = f"model.layers.{key.layer}.mlp.experts.exl3_{group}_{key.field}"
                param = params.get(target)
                if param is None:
                    raise ValueError(f"EXL3 expert target missing: {target}")
                param.weight_loader(param, tensor, expert_id=key.expert, projection=key.projection)
                acct.add("expert:native_trellis", tensor)
                continue
            if is_vision_qkv_duplicate(source_name):
                acct.add("dropped:vision_qkv_duplicate", tensor, 0)
                continue
            if config.dense_native.get(key.module):
                target = f"{strip_prefix(key.module)}.exl3_{key.field}"
                param = params.get(target)
                if param is None:
                    raise ValueError(f"EXL3 dense target missing: {target}")
                param.weight_loader(param, tensor)
                acct.add("dense:native_trellis", tensor)
                continue
            # sanctioned fallback: dense dequantize-on-load
            bucket = pending[key.module]
            bucket[key.field] = tensor
            if {"trellis", "suh", "svh", codebook} <= set(bucket):
                decoder = decoder or resolve_decoder()
                weight = dequantize(decoder, key.module, bucket, codebook)
                cls = _module_class(key.module)
                if cls not in logged:
                    log.warning("exl3_plain: dequantizing DENSE module class %s to BF16 (sanctioned fallback)", cls)
                    logged.add(cls)
                for t in bucket.values():
                    acct.add(f"dense:dequant->bf16 [{cls}]", t, 0)
                acct.rows[f"dense:dequant->bf16 [{cls}]"][2] += weight.numel() * 2
                remaining.append((key.module + ".weight", weight))
                del pending[key.module]
        if pending:
            raise ValueError(f"incomplete dense EXL3 modules at end of load: {sorted(pending)[:5]}")
        result = original(self, remaining, is_nextn=is_nextn)
        log.warning("exl3_plain byte accounting (end of load)\n%s", acct.report())
        return result
    return wrapped


def _quant_config(model):
    cfg = getattr(model, "quant_config", None)
    if cfg is None or cfg.get_name() != "exl3":
        raise ValueError("exl3_plain loader invoked without Exl3PlainConfig")
    return cfg


def _module_class(module: str) -> str:
    if module == "lm_head":
        return "lm_head"
    if module.startswith("model.visual"):
        return "vision." + module.split(".")[-1]
    parts = module.split(".")
    return ".".join(parts[4:]) if len(parts) > 4 else module
