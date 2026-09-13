"""exl3_plain v1: SGLang overlay for turboderp's plain EXL3 GLM-5.3-Flash branches (sm121, TP1)."""

from __future__ import annotations


def install() -> None:
    """Register ``exl3`` (plain contract) before ModelConfig parses the checkpoint."""
    from sglang.srt.layers.quantization import QUANTIZATION_METHODS
    from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration

    from .config import Exl3PlainConfig
    from .loader import wrap_load_weights

    registered = QUANTIZATION_METHODS.get("exl3")
    if registered is not None and registered is not Exl3PlainConfig:
        raise RuntimeError(f"'exl3' already registered to {registered!r}; the r6 selective overlay must not share an image")
    QUANTIZATION_METHODS["exl3"] = Exl3PlainConfig
    if not getattr(Glm5NextForConditionalGeneration.load_weights, "_exl3_plain", False):
        wrapped = wrap_load_weights(Glm5NextForConditionalGeneration.load_weights)
        wrapped._exl3_plain = True
        Glm5NextForConditionalGeneration.load_weights = wrapped
