# GLM-5.3-Flash on DGX Spark — one repo, every recipe

Everything needed to run **GLM-5.3-Flash on NVIDIA DGX Spark (GB10, 128 GB
unified, ARM64)** from one command per recipe, with receipts for every claim.

| Recipe | Stack | Weights | Quality (vs BF16 teacher, frozen 65,504-pos panel) | Status |
|---|---|---|---|---|
| [`recipes/turboderp-2p05-sglang-mul1/`](recipes/turboderp-2p05-sglang-mul1/) | **SGLang + sparkinfer(mul1+2bit) + exllamav3-ARM64** | turboderp 2.05bpw, 85.2 GB, pinned | **top-1 78.90 % · KL 0.384**; 262k retrieval 5/5; JSON-at-200k 5/5 | **serving, measured** (vision/MTP/speed: not claimed) |
| [`recipes/k2-tr3-2bpw-vllm/`](recipes/k2-tr3-2bpw-vllm/) | vLLM + B12x | 0xSero TR3 2.0bpw, 111.4 GB, pinned | top-1 77.39 % · KL 0.439 | prior release (superseded on quality/size; kept for reproducibility) |

Shared acceptance harnesses (loop, behavior, vision, long-context, speed)
live inside each recipe alongside their evidence so every recipe remains
independently verifiable. Quality panels and pre-registrations are frozen
and hashed inside each recipe's evidence.

## Quick start (new recipe)

```bash
cd recipes/turboderp-2p05-sglang-mul1
./fetch-weights.sh ../..//model          # one-time 85.2 GB pinned download
docker run --gpus all --ipc=host --shm-size 16g -p 8000:8000 \
  -v "$PWD/../../model:/model:ro" ghcr.io/0xsero/glm53-flash-exl3-plain@sha256:<see MEASURED.md>
```

## Credits

turboderp (EXL3 quants, MIT) · zai-org (GLM-5.3-Flash, MIT; cite arXiv:2602.15763)
· malaiwah (quant-fidelity measurement standard) · Luke Alonso (sparkinfer,
Apache-2.0) · NVIDIA (DGX Spark). See each recipe's README for details.
