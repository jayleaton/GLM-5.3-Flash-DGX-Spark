#!/usr/bin/env python3
"""GPU validation harness for the sparkinfer mul1 + 2-bit patches (MUL1-FEASIBILITY.md steps 2-4).

Steps, stop at first failure:
  a  synthetic dense K=3 mul1: patched sparkinfer trellis_linear vs exl3_reference (tolerance: fp16 pipeline)
  b  mcg regression: same synthetic through PRISTINE sparkinfer (subprocess) vs PATCHED (in-process), bit-identical
  c  oracle: exl3_reference.decode_tiles vs exllamav3 ext.reconstruct on real /model tensors, K in {2,3} (+5), bit-identical
  d  synthetic K=2 mul1 compile + decode, same tolerance as (a)

Exit codes: 0 pass | 2 usage | 10 environment/import | 20 step a | 30 step b | 40 step c | 43 oracle unavailable | 50 step d
Receipts: /out/validate-receipts.json (written on every exit path).
"""
from __future__ import annotations

import argparse, hashlib, json, os, subprocess, sys, time, traceback
from pathlib import Path

MARKER = {"mcg": 0xCBAC1FED, "mul1": 0x83DCD12D}
PATCHED_FILES = ("_lib/intrinsics.py", "moe/_shared/kernels/w4a16/kernel.py", "moe/_shared/kernels/w4a16/prepare.py",
                 "moe/_shared/execution.py", "moe/fused_moe/_impl.py")
# real dense modules for the oracle: (source module, expected class label); bits come from the shard header
ORACLE_MODULES = [
    "lm_head",
    "model.language_model.layers.3.self_attn.o_proj",
    "model.language_model.layers.30.self_attn.o_proj",
    "model.language_model.layers.0.mlp.gate_proj",
    "model.language_model.layers.5.mlp.shared_experts.down_proj",
    "model.language_model.layers.3.mlp.experts.0.gate_proj",
    "model.language_model.layers.44.mlp.experts.287.down_proj",
    "model.visual.blocks.0.mlp.up_proj",
    "model.visual.merger.proj",
]
ORACLE_PATTERNS = [".self_attn.qkv_proj", ".self_attn.q_a_proj", ".self_attn.kv_a_proj_with_mqa", ".self_attn.indexer.wq_b"]
TOL_ABS, TOL_RMS_REL = 2e-2, 2e-3   # fp16 rotation pipeline vs float32 reference; decode errors are O(1), not O(1e-2)


class Receipts:
    def __init__(self, path: Path):
        self.path, self.doc = path, {"schema": "exl3-plain-validate-v1", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "steps": [], "exit_code": None}

    def step(self, name: str, status: str, **fields):
        self.doc["steps"].append({"step": name, "status": status, **fields})
        self.flush()

    def flush(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.doc, indent=1, default=str))

    def exit(self, code: int, why: str = "") -> int:
        self.doc["exit_code"], self.doc["exit_reason"], self.doc["finished_utc"] = code, why, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.flush()
        print(f"validate_mul1: exit {code} {why}")
        return code


def marker_tensor(codebook: str, device) -> "torch.Tensor":
    import torch
    v = MARKER[codebook]
    return torch.tensor([v - (1 << 32) if v >= 1 << 31 else v], dtype=torch.int32, device=device)


def synthetic_tile(bits: int, k: int, n: int, seed: int):
    import torch
    g = torch.Generator().manual_seed(seed)
    trellis = torch.randint(-32768, 32767, (k // 16, n // 16, 16 * bits), generator=g, dtype=torch.int16)
    suh = torch.ones(k, dtype=torch.float16)
    svh = torch.ones(n, dtype=torch.float16)
    return trellis, suh, svh


def run_sparkinfer_dense(trellis, suh, svh, codebook: str, m_rows: int):
    """y = trellis_linear.run(I_rows) on the current sparkinfer tree; returns fp16 [M, N] on CPU."""
    import torch
    from sparkinfer.gemm import trellis_linear
    dev = torch.device("cuda")
    w = trellis_linear.prepare_weight(trellis.to(dev), suh.to(dev), svh.to(dev), **{codebook: marker_tensor(codebook, dev)}, params_dtype=torch.float16)
    k = trellis.shape[0] * 16
    x = torch.eye(m_rows, k, dtype=torch.float16, device=dev)
    y = trellis_linear.run(x, w)
    torch.cuda.synchronize()
    return y.to(torch.float16).cpu()


def reference_dense(trellis, suh, svh, codebook: str, m_rows: int):
    """Same product in the reference: H_K W̃ H_N (suh=svh=1) rows 0..M-1, fp16 with float32 math."""
    import torch
    from exl3_plain_sglang_overlay import exl3_reference as R
    inner = R.decode_tiles(trellis, codebook)
    return R.outer_weight(inner, suh, svh)[:m_rows]


def compare(y, y_ref):
    import torch
    d = (y.float() - y_ref.float())
    scale = y_ref.float().abs().max().item() or 1.0
    return {"max_abs": d.abs().max().item(), "rms_rel": (d.pow(2).mean().sqrt() / (y_ref.float().pow(2).mean().sqrt() + 1e-12)).item(),
            "ref_max": scale, "bit_identical": bool(torch.equal(y.to(torch.float16), y_ref.to(torch.float16)))}


def within(m) -> bool:
    return m["max_abs"] <= TOL_ABS * max(1.0, m["ref_max"]) and m["rms_rel"] <= TOL_RMS_REL


# ---------------------------------------------------------------- worker (runs under the pristine tree)
def worker(args) -> int:
    import torch
    trellis, suh, svh = synthetic_tile(args.bits, args.k, args.n, args.seed)
    y = run_sparkinfer_dense(trellis, suh, svh, args.codebook, args.rows)
    import sparkinfer
    torch.save({"y": y, "sparkinfer_file": sparkinfer.__file__, "inputs_sha": hashlib.sha256(trellis.numpy().tobytes()).hexdigest()}, args.save)
    return 0


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="/model")
    ap.add_argument("--out", default="/out/validate-receipts.json")
    ap.add_argument("--pristine", default="/opt/sparkinfer-pristine", help="dir containing an UNPATCHED sparkinfer/ package")
    ap.add_argument("--k", type=int, default=256); ap.add_argument("--n", type=int, default=256); ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--skip-oracle", action="store_true", help="treat missing exllamav3 as SKIP instead of exit 43 (still records)")
    # worker mode
    ap.add_argument("--worker", choices=["synthetic"]); ap.add_argument("--codebook", default="mcg"); ap.add_argument("--bits", type=int, default=3); ap.add_argument("--save")
    args = ap.parse_args(argv)
    if args.worker:
        return worker(args)

    rc = Receipts(Path(args.out))
    # ---- environment
    try:
        import torch, sparkinfer
        from sparkinfer.gemm import trellis_linear
        from sparkinfer.moe import fused_moe
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from exl3_plain_sglang_overlay import exl3_reference as R
        assert torch.cuda.is_available(), "no CUDA device"
        root = Path(sparkinfer.__file__).parent
        pristine_root = Path(args.pristine) / "sparkinfer"
        hashes = {f: {"patched": hashlib.sha256((root / f).read_bytes()).hexdigest()[:16],
                      "pristine": hashlib.sha256((pristine_root / f).read_bytes()).hexdigest()[:16] if (pristine_root / f).is_file() else None} for f in PATCHED_FILES}
        patched = all(h["pristine"] and h["patched"] != h["pristine"] for h in hashes.values())
        rc.doc["env"] = {"torch": torch.__version__, "device": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0),
                         "sparkinfer": str(root), "pristine": str(pristine_root), "patched_files": hashes, "tree_is_patched": patched,
                         "trellis_linear_supported": bool(trellis_linear.is_supported()), "fused_moe_supported": bool(fused_moe.is_supported())}
        rc.flush()
        if not patched:
            return rc.exit(10, "sparkinfer tree at import path is not the patched one (or pristine copy missing)")
        if not trellis_linear.is_supported():
            return rc.exit(10, "trellis_linear.is_supported() is False on this device")
    except Exception as e:
        rc.step("env", "FAIL", error=repr(e), trace=traceback.format_exc())
        return rc.exit(10, f"environment: {e!r}")
    os.environ.setdefault("SPARKINFER_PRINT_COMPILE_PROGRESS", "1")

    # ---- step a: synthetic K=3 mul1
    t0 = time.time()
    try:
        tr, suh, svh = synthetic_tile(3, args.k, args.n, args.seed)
        y = run_sparkinfer_dense(tr, suh, svh, "mul1", args.rows)
        y_ref = reference_dense(tr, suh, svh, "mul1", args.rows)
        m = compare(y, y_ref)
        ok = within(m)
        rc.step("a_synthetic_k3_mul1", "PASS" if ok else "FAIL", bits=3, codebook="mul1", shape=[args.k, args.n], seconds=round(time.time() - t0, 1), **m)
        if not ok:
            return rc.exit(20, f"step a: mul1 K=3 kernel vs reference outside tolerance {m}")
    except Exception as e:
        rc.step("a_synthetic_k3_mul1", "FAIL", error=repr(e), trace=traceback.format_exc(), seconds=round(time.time() - t0, 1))
        return rc.exit(20, f"step a raised: {e!r}")

    # ---- step b: mcg regression pristine vs patched
    t0 = time.time()
    try:
        tr, suh, svh = synthetic_tile(3, args.k, args.n, args.seed + 1)
        y_patched = run_sparkinfer_dense(tr, suh, svh, "mcg", args.rows)
        save = Path(args.out).parent / "mcg-pristine.pt"
        env = dict(os.environ, PYTHONPATH=f"{args.pristine}:{Path(__file__).resolve().parent}")
        cmd = [sys.executable, __file__, "--worker", "synthetic", "--codebook", "mcg", "--bits", "3", "--k", str(args.k), "--n", str(args.n),
               "--rows", str(args.rows), "--seed", str(args.seed + 1), "--save", str(save)]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
        if proc.returncode != 0:
            rc.step("b_mcg_regression", "FAIL", error="pristine worker failed", stdout=proc.stdout[-2000:], stderr=proc.stderr[-4000:])
            return rc.exit(30, "step b: pristine worker failed")
        blob = torch.load(save)
        if not blob["sparkinfer_file"].startswith(args.pristine):
            rc.step("b_mcg_regression", "FAIL", error=f"worker imported {blob['sparkinfer_file']}, not the pristine tree")
            return rc.exit(30, "step b: worker did not import the pristine tree")
        m = compare(y_patched, blob["y"])
        m_ref = compare(y_patched, reference_dense(tr, suh, svh, "mcg", args.rows))
        ok = m["bit_identical"]
        rc.step("b_mcg_regression", "PASS" if ok else "FAIL", bits=3, codebook="mcg", pristine_file=blob["sparkinfer_file"],
                pristine_vs_patched=m, patched_vs_reference=m_ref, seconds=round(time.time() - t0, 1))
        if not ok:
            return rc.exit(30, f"step b: mcg output changed by the patch {m}")
    except Exception as e:
        rc.step("b_mcg_regression", "FAIL", error=repr(e), trace=traceback.format_exc(), seconds=round(time.time() - t0, 1))
        return rc.exit(30, f"step b raised: {e!r}")

    # ---- step c: oracle on real tensors
    t0 = time.time()
    try:
        from exllamav3.ext import exllamav3_ext as ext  # noqa
        have_oracle = True
    except Exception as e:
        have_oracle, oracle_err = False, repr(e)
    if not have_oracle:
        rc.step("c_oracle_real_tensors", "SKIP" if args.skip_oracle else "FAIL", error=f"exllamav3 unavailable: {oracle_err}")
        if not args.skip_oracle:
            return rc.exit(43, "step c: exllamav3 oracle unavailable in this image (pip install exllamav3 / build ext for sm121)")
    else:
        try:
            from safetensors import safe_open
            index = json.loads((Path(args.model) / "model.safetensors.index.json").read_text())["weight_map"]
            modules = list(ORACLE_MODULES)
            for pat in ORACLE_PATTERNS:  # first layer that has each attention class
                cand = sorted({k.rsplit(".", 1)[0] for k in index if pat in k and k.endswith(".trellis")}, key=lambda s: int(s.split(".layers.")[1].split(".")[0]))
                if cand:
                    modules.append(cand[0])
            results, bits_seen = [], set()
            for mod in modules:
                names = {f: f"{mod}.{f}" for f in ("trellis", "suh", "svh", "mul1", "mcg")}
                if names["trellis"] not in index:
                    results.append({"module": mod, "status": "MISSING"}); continue
                codebook = "mul1" if names["mul1"] in index else "mcg"
                def load(n):
                    with safe_open(str(Path(args.model) / index[n]), "pt", device="cpu") as f:
                        return f.get_tensor(n)
                trellis = load(names["trellis"])
                bits = trellis.shape[-1] // 16
                kb, nb = min(trellis.shape[0], 64), min(trellis.shape[1], 64)   # ≤1024x1024 sub-block, tiles are independent
                sub = trellis[:kb, :nb].contiguous()
                ref = R.decode_tiles(sub, codebook)
                dev = torch.device("cuda")
                unpacked = torch.empty(kb * 16, nb * 16, dtype=torch.float16, device=dev)
                ext.reconstruct(unpacked, sub.to(dev), bits, codebook == "mcg", codebook == "mul1")
                torch.cuda.synchronize()
                m = compare(unpacked.cpu(), ref)
                bits_seen.add(bits)
                results.append({"module": mod, "bits": bits, "codebook": codebook, "sub_block": [kb * 16, nb * 16], "full_shape": [trellis.shape[0] * 16, trellis.shape[1] * 16], **m})
                if not m["bit_identical"]:
                    rc.step("c_oracle_real_tensors", "FAIL", results=results, seconds=round(time.time() - t0, 1))
                    return rc.exit(40, f"step c: reference decoder differs from exllamav3 reconstruct on {mod} (K={bits}) {m}")
            missing = {2, 3} - bits_seen
            status = "PASS" if not missing else "FAIL"
            rc.step("c_oracle_real_tensors", status, bits_covered=sorted(bits_seen), results=results, seconds=round(time.time() - t0, 1))
            if missing:
                return rc.exit(40, f"step c: real tensors did not cover K={sorted(missing)}; extend ORACLE_MODULES")
        except Exception as e:
            import pathlib
            if args.skip_oracle and not (pathlib.Path(args.model) / "model.safetensors.index.json").is_file():
                rc.step("c_oracle_real_tensors", "SKIP", error=f"model absent on this node, skip-oracle run: {e!r}")
            else:
                rc.step("c_oracle_real_tensors", "FAIL", error=repr(e), trace=traceback.format_exc(), seconds=round(time.time() - t0, 1))
                return rc.exit(40, f"step c raised: {e!r}")

    # ---- step d: synthetic K=2 mul1 compile + decode
    t0 = time.time()
    try:
        tr, suh, svh = synthetic_tile(2, args.k, args.n, args.seed + 2)
        y = run_sparkinfer_dense(tr, suh, svh, "mul1", args.rows)
        m = compare(y, reference_dense(tr, suh, svh, "mul1", args.rows))
        ok = within(m)
        rc.step("d_synthetic_k2_mul1", "PASS" if ok else "FAIL", bits=2, codebook="mul1", shape=[args.k, args.n], seconds=round(time.time() - t0, 1), **m)
        if not ok:
            return rc.exit(50, f"step d: mul1 K=2 kernel vs reference outside tolerance {m}")
    except Exception as e:
        rc.step("d_synthetic_k2_mul1", "FAIL", error=repr(e), trace=traceback.format_exc(), seconds=round(time.time() - t0, 1))
        return rc.exit(50, f"step d raised: {e!r}")

    return rc.exit(0, "all steps passed")


if __name__ == "__main__":
    sys.exit(main())
