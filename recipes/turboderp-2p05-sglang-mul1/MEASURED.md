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

## Not claimed

Vision/video acceptance (tower loads; case triage in progress). MTP/speculative
decoding. Speed-cell benchmarks (pending). Any panel besides the frozen one.
