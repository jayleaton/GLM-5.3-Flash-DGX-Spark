# GLM-5.3-Flash on DGX Spark

Run GLM-5.3-Flash (320B params, 18B active) on a single NVIDIA DGX Spark with one Docker command. Every recipe includes measured quality numbers against the original BF16 model.

## What's here

Two ways to run the model, both tested on real hardware:

| | Recipe A (recommended) | Recipe B (prior release) |
|---|---|---|
| **Path** | [`recipes/turboderp-2p05-sglang-mul1/`](recipes/turboderp-2p05-sglang-mul1/) | [`recipes/k2-tr3-2bpw-vllm/`](recipes/k2-tr3-2bpw-vllm/) |
| **Engine** | SGLang | vLLM |
| **Weights** | turboderp 2.05bpw (85 GB) | 0xSero TR3 2.0bpw (111 GB) |
| **Quality** | 78.9% match · KL 0.384 | 77.4% match · KL 0.439 |
| **Context** | 262,144 tokens | 204,800 tokens |

Recipe A is smaller, faster to download, closer to the original model, and supports full context. Recipe B is kept for reproducibility.

## Quick start

**Prerequisites:** one DGX Spark (128 GB), Docker, ~85 GB disk for weights.

```bash
# 1. Download the weights (one time, ~85 GB)
git clone https://github.com/0xSero/GLM-5.3-Flash-DGX-Spark.git
cd GLM-5.3-Flash-DGX-Spark/recipes/turboderp-2p05-sglang-mul1
./fetch-weights.sh ./model

# 2. Start the server
docker run --gpus all --ipc=host --shm-size 16g -p 8000:8000 \
  -v "$PWD/model:/model:ro" \
  ghcr.io/0xsero/glm53-flash-exl3-plain@sha256:85cb3fa86d31a781b94dcf10ee168adf096cfeaac14d2f1e6c560504e58e4eed

# 3. Use it
curl http://localhost:8000/v1/models
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/model","messages":[{"role":"user","content":"Hello"}]}'
```

## What's measured

Every claim has a receipt in [`recipes/turboderp-2p05-sglang-mul1/MEASURED.md`](recipes/turboderp-2p05-sglang-mul1/MEASURED.md):

- **Quality:** 78.9% top-1 agreement with the BF16 original across 65,504 test positions (KL divergence 0.384)
- **Long context:** correctly retrieves facts from 262,016-token prompts
- **Structured output:** generates valid JSON 1-300 at every context length up to 200k
- **Pre-registered benchmark:** 15/20 tasks pass (retrieval 5/5, structured output 5/5, reasoning 4/5)
- **Vision:** 6/6 synthetic image/video tests pass
- **No looping:** 5/5 repetition-safety checks pass

**Not yet claimed:** MTP/speculative decoding, speed benchmarks with CUDA graphs, full vision evaluation.

## How Recipe A works

The stock kernels couldn't run this model. Three things were built and validated:

1. **mul1 codebook + 2-bit support in sparkinfer** (`sparkinfer-patches/`) — the trellis decoder only spoke "mcg" at 3-6 bits; now it handles the "mul1" codebook at 2-6 bits, verified bit-exact against exllamav3
2. **exllamav3 ARM64 port** (`build/exllamav3-arm64/`) — the reference runtime didn't compile on ARM; now it does (CPU expert offload and native-TP all-reduce are the only losses)
3. **SGLang adapter** (`adapter/`) — nine patches that teach SGLang's `exl3` quantization method to load and serve the artifact, including fixes for two silent-corruption bugs (skipped KDA conv weights, unscaled MoE routing)

Each patch is in `build/*.diff` with a plain-language description in the recipe README.

## Credits

- **[turboderp](https://huggingface.co/turboderp)** — the EXL3 quantizations (MIT)
- **[zai-org](https://huggingface.co/zai-org)** — GLM-5.3-Flash (MIT); cite [arXiv:2602.15763](https://arxiv.org/abs/2602.15763)
- **[malaiwah](https://github.com/malaiwah/quant-fidelity-suite)** — the quality measurement standard
- **[Luke Alonso](https://github.com/local-inference-lab)** — sparkinfer (Apache-2.0)
- **NVIDIA** — DGX Spark hardware

MIT license. See [LICENSE](LICENSE).
