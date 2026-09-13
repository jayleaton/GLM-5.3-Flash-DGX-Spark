# GLM-5.3-Flash · turboderp 2.05bpw EXL3 · SGLang on one DGX Spark (mul1 stack)

Serve [`turboderp/GLM-5.3-Flash-exl3`](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3)
(branch `2.05bpw`, pinned revision `51058cd551c7e570d87bd32a4adee720edce2349`)
on **one NVIDIA DGX Spark (GB10, ARM64, sm121)** under **SGLang** with the
full 262,144-token context, images/video enabled at the API level, and MTP
layers loaded but speculation off by default.

This recipe exists because the stock kernels could not run this artifact:
sparkinfer had no `mul1` codebook and no 2-bit trellis path, exllamav3 had no
ARM64 build, and SGLang had no `exl3` quantization method. All three were
solved and validated; see **[MEASURED.md](MEASURED.md)** for receipts.

## Status (2026-09-13)

| Capability | Status |
|---|---|
| Coherent generation (long-form, greedy) | **verified live** (correct arithmetic, structured reasoning, natural stops) |
| Context | **262,144 configured; 262,016-token retrieval test passes** (5/5 retrieval family incl. one at 262,016) |
| Quality vs BF16 teacher (65,504-pos frozen panel) | **top-1 78.90 % · KL 0.384** — above the prior K2 release (77.39 % / 0.439) at 85.2 GB vs 111.4 GB |
| 20 pre-registered samples | **15/20** (structured decode 5/5 · retrieval 5/5 · reasoning 4/5 · free-form generation 1/5 — weak family, under triage) |
| Vision / video | **experimental, not yet claimed** — tower loads and serves; acceptance cases still being triaged |
| MTP / speculative | **off by default, not claimed** |
| CUDA graphs | works for decode; prefill graphs auto-disabled with KDA attention (upstream behavior) |
| Speed cells | pending |

## One-command run

Weights (85.2 GB) are pulled once from the pinned revision into `./model`
(see `fetch-weights.sh`), then:

```bash
docker run --gpus all --ipc=host --shm-size 16g -p 8000:8000 \
  -v "$PWD/model:/model:ro" \
  ghcr.io/0xsero/glm53-flash-exl3-plain@sha256:<digest-pinned-below> \
  # serving flags baked into the image ENTRYPOINT; see MEASURED.md §launch
curl http://127.0.0.1:8000/v1/models
```

The exact digest and flag set are recorded in **MEASURED.md**. If you prefer
to build the image yourself, `build/Dockerfile.serve` applies all nine
patches (`build/*.diff`) to the pinned public base image — no private layers.

## What the nine patches do

`sparkinfer-patches/` — kernel: adds the `mul1` trellis codebook (a 12-instruction
hash variant) to the CuTe-DSL decode kernel and widens the bit range to 2–6.
`exllamav3-arm64/` — build port: guards the x86-only AVX/CPU-MoE translation
units so the CUDA extension compiles on aarch64 (losses: CPU expert offload and
the native-TP CPU all-reduce only; TP1 unaffected).
`build/` — the SGLang overlay (`adapter/`) plus: contract widening (mul1 +
2-bit), fused-linear fallback (fused SGLang linears dequantize to BF16,
announced + byte-accounted), dtype hardening, kernel codebook binding,
CUDA-graph storage for dense trellis, the KDA conv1d weight split, the MoE
routed-scaling fix, and a completeness gate that aborts loading if any
checkpoint tensor goes unconsumed. Every patch was individually validated —
see VALIDATE-RUN.md and GPU-GATES.md.

## Credits

- **turboderp** — the EXL3 quantization and the artifact (MIT); quantization author.
- **zai-org** — GLM-5.3-Flash (MIT). Citation requested: *GLM-5: from Vibe
  Coding to Agentic Engineering* (arXiv:2602.15763).
- **malaiwah** — the quant-fidelity measurement standard this work's receipts follow.
- **Luke Alonso** — sparkinfer (Apache-2.0).

No credentials, private addresses, or unreproducible steps are required.
