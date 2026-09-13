# VALIDATE-RUN — mul1 / 2-bit sparkinfer patches on 557f

1. Build the base once: `docker build -t glm53-flash-sglang-exl3-plain:v1 .` (de5c or 557f, ARM64).
2. Build the validation image: `docker build -f Dockerfile.mul1 -t glm53-flash-sglang-exl3-plain:mul1-validate .` — applies `patches/000{1,2,3}` to a copy in site-packages, keeps the pristine tree at `/opt/sparkinfer-pristine`, installs exllamav3 (oracle).
3. Run: `MODEL=<path-to-model-dir>; mkdir -p ./out`
4. `docker run --rm --gpus all --ipc=host -v "$MODEL":/model:ro -v "$PWD/out":/out -e PYTHONPATH=/w2port -e SPARKINFER_PRINT_COMPILE_PROGRESS=1 glm53-flash-sglang-exl3-plain:mul1-validate python3 /w2port/validate_mul1.py`
5. Steps run in order a → b → c → d and stop at the first failure; receipts land in `out/validate-receipts.json` on every exit path (plus `out/mcg-pristine.pt`).
6. Exit codes: 0 all pass · 10 environment (unpatched tree, no CUDA, import) · 20 step a · 30 step b · 40 step c · 43 exllamav3 oracle missing · 50 step d.
7. First run JIT-compiles the K=3 mul1, K=3 mcg (twice: pristine + patched), and K=2 mul1 kernels; expect minutes per compile, printed by `SPARKINFER_PRINT_COMPILE_PROGRESS`.
8. If exllamav3 fails to build for sm121, rerun with `--skip-oracle` to still get a/b/d, but gate 2b stays open until step c passes.
9. Tolerances: a/d compare the kernel's fp16 rotation pipeline to the float32 reference (`max_abs ≤ 2e-2·max|y|`, `rms_rel ≤ 2e-3`); b and c require bit-identity.
10. Attach `validate-receipts.json` to the gate 1/2b lines of `GATES.md`; only after exit 0 widen `contract.py` `SPARKINFER_CODEBOOKS` / `SPARKINFER_TRELLIS_BITS`.
