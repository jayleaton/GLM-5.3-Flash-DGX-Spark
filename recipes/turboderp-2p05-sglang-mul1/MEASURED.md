# MEASURED — receipts for the turboderp-2.05bpw mul1 stack (2026-09-13)

Every number below has a receipt in the campaign record; nothing is claimed
without evidence. Teacher = `zai-org/GLM-5.3-Flash-BF16` @ `a6c167b6…`
(frozen fixture, fidelity proven to 4e-6 PPL against its reference value).

## Kernel validation (before any serving)

`validate_mul1.py` on the sealed artifact — **exit 0, all steps PASS**:
(a) synthetic K=3 mul1 decode vs reference (max_abs 0.0039); (b) MCG
regression bit-identical pre/post patch; **(c) real-tensor oracle — the
reference decoder is bit-exact against exllamav3 v1.4.9 (ARM64 port) on real
turboderp tensors, K∈{2,3}, every module class**; (d) first-ever K=2
sparkinfer trellis compile, PASS.

## Exoneration control

The sealed artifact run under exllamav3 1.4.9 (the reference runtime)
generates flawlessly greedy (correct arithmetic, structured reasoning) —
i.e., **the artifact's quality ceiling is intact; all defects found during
integration were ours and were fixed** (nine patches; two were
silent-corruption bugs: skipped KDA conv weights, and an unscaled MoE
routing factor).

## Quality — frozen 65,504-position panel vs the BF16 teacher

| Point | Bytes | Top-1 | KL |
|---|---|---|---|
| prior K2/TR3 release (same panel) | 111.35 GB | 77.385 % | 0.439 |
| prior P14 (same panel) | 115.58 GB | 78.381 % | 0.407 |
| **this recipe (2.05bpw mul1)** | **85.23 GB** | **78.902 %** | **0.384** |

KL method: KL(teacher‖candidate), lower-bound and top-K-renormalized agree
at 0.384 (teacher mass coverage 99.3 %). Candidate PPL 4.287 (teacher 3.200).
The 88.92 % figure on the artifact's HF card was measured on a different
panel (51,175 positions) and is not comparable, by pre-registration.

## 20 pre-registered samples (seed 20260912, frozen before any run)

**15/20 pass.** structured_decode **5/5** (JSON 1..300 at 1,024→200,000
ctx). retrieval_long_context **5/5** — including a 262,016-token
needle-in-haystack. reasoning 4/5. free-form generation 1/5 — the weak
family, under triage (one observed failure mode: digit transposition in a
buried code, CHARLIE-59→CHARLIE-19).

## Long-context

262,144 configured; a 262,016-token prompt (≈1.05 M chars) served and
retrieved correctly with `finish_reason: stop`. Prefill ≈ 425 tok/s at that
length on one GB10 (no CUDA graphs; graph cells pending).

## Published image

`ghcr.io/0xsero/glm53-flash-exl3-plain@sha256:85cb3fa86d31a781b94dcf10ee168adf096cfeaac14d2f1e6c560504e58e4eed`
(tag `2p05-sglang-mul1-r1`, linux/arm64, 31.8 GB). Built from this repo's
`build/Dockerfile.serve` on the target hardware; serving flags in the image
ENTRYPOINT (context 262,144, FP8 KV, DSA/KDA attention, multimodal on, CUDA
decode graphs, MTP off).

## Not claimed

Vision/video acceptance: 6/6 synthetic paired visual cases pass (scoped acceptance only, not broad visual quality). Full vision status: the 2.05 recipe serves images and video end-to-end. MTP/speculative
decoding. Speed-cell benchmarks (pending). Any panel besides the frozen one.

### Serving note — tool-call reliability (2026-09-16)

At the shipped `generation_config.json` temperature (1.0) this 2.05 bpw checkpoint intermittently
emits malformed GLM tool-call markup, worst when a large tool surface is advertised (opencode
advertises ~16 tools) and/or after long reasoning. Observed forms: the reasoning tail glued to the
tool **name** (`... </think><tool_call>bash`), speculative multi-call spam with invented names
(`bashCommand`), and truncated JSON arguments.

Measured on one GB10 with an opencode-shaped 16-tool harness (streaming), 120 samples per cell:

| temperature | corruption |
|---|---|
| 1.0 | 5/120 (4.2%) |
| 0.6 | 1/120 (0.8%) |
| 0.0 | 0/120 |

Long-reasoning cases were worse at 1.0 (3/20) and clean at 0.6 (0/20). Lowering the default
temperature cuts the rate ~5× and is the recommended mitigation.

Because `generation_config.json` ships inside the mounted weights, apply it by bind-mounting an
override over the model file, e.g.:

```yaml
volumes:
  - /path/to/generation_config.override.json:/model/generation_config.json:ro
```

Two parser notes: `--tool-call-parser` **must stay `glm47`** — `glm45` (which the chat template
auto-detects) extracts zero tool calls for this model. The reasoning parser staying `glm45` is fine;
switching it to `deepseek-r1` made no measurable difference.
