# GPU gate checklist — exl3_plain v1 (557f, sealed 2.05bpw dir)

Mark each line with UTC time + receipt path. Never skip a line; a failing
line stops the ladder. "Loud" means the process exits non-zero with the
adapter's message — that is a pass for the *check*, a fail for the *gate*.

## Gate 0 — build (de5c)
- [ ] `docker build` succeeds; digest pinned in the receipt.
- [ ] Build-time assertions all printed: torch ≥ 2.12, cutlass-dsl 4.6.0, `sparkinfer import ok`, `overlay import ok`, index-only census exit 0.

## Gate 1 — import smoke (557f, GPU)
- [ ] `python3 -c "import sparkinfer.moe.fused_moe as m, sparkinfer.gemm.trellis_linear as t; print(m.is_supported(), t.is_supported())"` → `True True` on sm121.
- [ ] One tilelang/triton kernel JIT-compiles for sm121a: run `SPARKINFER_PRINT_COMPILE_PROGRESS=1` with a tiny `trellis_linear.prepare_weight`/`run` on a synthetic MCG 4-bit tensor (K=N=256). Log line shows the compile and its duration.
- [ ] Full census (`census.py /model`) exit 0; receipt saved; `codebook`, `experts.bits_histogram`, `dense.*.bits` recorded.

## Gate 2 — full weight load, zero unaccounted bytes
- [ ] **Current status: BLOCKED (OPEN-QUESTIONS Q10).** Expected exit: `STOP: routed experts cannot be served natively by sparkinfer 1.0.1 ... Census reasons: [codebook 'mul1' ..., expert trellis bits [...] ...]` at `layers.3.mlp.experts`, before any shard is read. Record the exact message; do not patch around it.
- [ ] (once unblocked) Load completes; end-of-load byte report printed; every row is one of `expert:native_trellis`, `dense:native_trellis`, `dense:dequant->bf16 [class]`, `bf16:passthrough`, `dropped:vision_qkv_duplicate`, `dropped:nextn_mtp_off`. TOTAL tensors == 151,554.
- [ ] `dropped:vision_qkv_duplicate` == 360 tensors (24 blocks × (3 proj × 4 fields + 3 biases)); `dropped:nextn_mtp_off` == 3,508 tensors (3,456 expert + 36 dense + 16 BF16).
- [ ] Every `dense:dequant->bf16` class was announced by one `sanctioned fallback` log line; resident GB matches N×K×2 per module.
- [ ] Resident weights (from the byte report) + 262,144-ctx FP8 KV (2.0–2.2 GiB) + graphs ≤ the ~119 GiB admission budget; record for `CAMPAIGN-12H.md` M0.

## Gate 2b — decoder equivalence (required before any quality number)
- [ ] For ≥ 8 dense modules spanning every class (o_proj, qkv_proj, q_a/q_b/kv_a, indexer.wq_b, dense MLP, shared_experts, lm_head, vision mlp/merger/attn.proj): `decode(...)` output vs exllamav3's own `Linear` forward on 16 random BF16 inputs → bit-identical logits contribution or max abs diff ≤ 1e-3 with the cause recorded. Receipt lists module, bits, marker, max diff.

## Gate 3 — CUDA graphs
- [ ] Decode graphs captured for bs ladder ≤ 8 (`--cuda-graph-max-bs-decode 8`); prefill graph bs=1 only (`--cuda-graph-max-bs-prefill 1`). Log shows capture count and no `KernelResolutionFrozenError` after `freeze_kernel_resolution` if enabled.
- [ ] Replay check: same prompt twice, identical token IDs.
- [ ] A 1025-token prefill chunk is refused loudly (batch > `SGLANG_EXL3_MAX_BATCH_TOKENS`), never silently split into eager.

## Gate 4 — endpoint
- [ ] `GET /v1/models` → 200 with the model id.
- [ ] One chat completion, 64 new tokens, `finish_reason: stop`, coherent text.
- [ ] Two cold starts (N≥2) reach this line; times recorded.

## Gate 5 — campaign gates (CAMPAIGN-12H.md §2)
- [ ] Vision 4/4 image cases; video 2/2.
- [ ] 262,016-token request, 3 retrieval codes correct, no OOM; resident bytes recorded (G3).
- [ ] Loop/degeneration harness on exact token IDs: pass.
- [ ] KLD panel (65,504 positions) vs BF16 teacher `a6c167b6…` + QFS row; 20 pre-registered samples (G4).
- [ ] C1 speed cells at the final config (G6), MTP off.

## Gate 6 — NEXTN follow-up (separate experiment; see LAUNCH.md §5)
- [ ] Not started. Requires the NEXTN worker path; MTP stays off until G3-MTP passes.
