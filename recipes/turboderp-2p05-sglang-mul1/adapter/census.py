"""Census of a plain EXL3 artifact: the build-time / launch-time assertion.

Reads only the safetensors index, every shard header, and the 4-byte codebook
markers.  Never touches trellis bytes.  Writes a JSON receipt that
``Exl3PlainConfig`` requires (``EXL3_PLAIN_CENSUS``) before SGLang may load.

    python -m exl3_plain_sglang_overlay.census /model --out census.json
    python -m exl3_plain_sglang_overlay.census --index-only path/to/model.safetensors.index.json

Exit 0 only when the contract holds.  The verdict says, per module class,
whether sparkinfer can execute it natively; a false verdict is a finding,
not an error, and the adapter fails closed on it at load.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import mmap
import re
import struct
import sys
from pathlib import Path

from .contract import (
    CODEBOOK_SENTINELS, EXPERTS, EXPERT_PROJECTIONS, FIELDS, FIRST_MOE_LAYER, LAST_MOE_LAYER,
    MARKER_FIELDS, NEXTN_LAYER, SPARKINFER_CODEBOOKS, SPARKINFER_TRELLIS_BITS,
    is_vision_qkv_duplicate, parse, trellis_bits,
)

SCHEMA = "exl3-plain-census-v1"
_NUM = re.compile(r"\.(\d+)\.")


def _pattern(name: str) -> str:
    return _NUM.sub(lambda m: ".{N}.", name)


def _read_headers(root: Path, shards: list[str]) -> dict[str, dict]:
    """tensor name -> {dtype, shape, marker?} from every shard header."""
    out: dict[str, dict] = {}
    for shard in shards:
        path = root / shard
        with open(path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            (hlen,) = struct.unpack("<Q", mm[:8])
            header = json.loads(mm[8:8 + hlen])
            base = 8 + hlen
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                entry = {"dtype": meta["dtype"], "shape": tuple(meta["shape"]), "shard": shard}
                if name.rsplit(".", 1)[-1] in MARKER_FIELDS:
                    start, end = meta["data_offsets"]
                    if end - start != 4:
                        raise SystemExit(f"census: marker {name} is {end - start} bytes, expected 4")
                    entry["marker"] = struct.unpack("<I", mm[base + start: base + end])[0]
                out[name] = entry
    return out


def census(index_path: Path, *, index_only: bool) -> dict:
    index = json.loads(index_path.read_text())
    weight_map: dict[str, str] = index["weight_map"]
    root = index_path.parent
    shards = sorted(set(weight_map.values()))
    headers = {} if index_only else _read_headers(root, shards)
    if not index_only and set(headers) != set(weight_map):
        missing = sorted(set(weight_map) - set(headers))[:5]
        extra = sorted(set(headers) - set(weight_map))[:5]
        raise SystemExit(f"census: index/shard mismatch; missing={missing} extra={extra}")

    patterns = collections.Counter(_pattern(n) for n in weight_map)
    fields = collections.Counter()
    unknown_fields: list[str] = []
    expert_layers: dict[int, set[int]] = collections.defaultdict(set)
    expert_bits: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    dense: dict[str, dict] = {}
    markers = collections.Counter()
    vision_dups = set()
    problems: list[str] = []

    for name in weight_map:
        key = parse(name)
        if key is None:
            continue  # BF16 passthrough (weights, biases, hc_*, A_log, dt_bias, ...)
        fields[key.field] += 1
        if key.field not in FIELDS:
            unknown_fields.append(name)
        hdr = headers.get(name)
        if key.field in MARKER_FIELDS and hdr is not None:
            codebook = CODEBOOK_SENTINELS.get(hdr["marker"])
            if codebook != key.field:
                problems.append(f"{name}: marker {hdr['marker']:#010x} is not the {key.field} sentinel")
            markers[f"{key.field}:{hdr['marker']:#010x}"] += 1
        if key.kind == "expert":
            expert_layers[key.layer].add(key.expert)
            if key.field == "trellis" and hdr is not None:
                expert_bits[key.layer][trellis_bits(hdr["shape"], hdr["dtype"])] += 1
            continue
        if is_vision_qkv_duplicate(name):
            vision_dups.add(key.module)
        entry = dense.setdefault(key.module, {"fields": [], "layer": key.layer})
        entry["fields"].append(key.field)
        if key.field == "trellis" and hdr is not None:
            bits = trellis_bits(hdr["shape"], hdr["dtype"])
            k, n = hdr["shape"][-3] * 16, hdr["shape"][-2] * 16
            entry.update(bits=bits, in_features=k, out_features=n,
                         dims_div_128=(k % 128 == 0 and n % 128 == 0))

    # --- contract assertions -------------------------------------------------
    for layer, ids in sorted(expert_layers.items()):
        if ids != set(range(EXPERTS)):
            problems.append(f"layer {layer}: {len(ids)} experts, expected {EXPERTS}")
    expected_layers = set(range(FIRST_MOE_LAYER, LAST_MOE_LAYER + 1)) | {NEXTN_LAYER}
    if set(expert_layers) != expected_layers:
        problems.append(f"expert layers {sorted(expert_layers)} != {sorted(expected_layers)}")
    for module, entry in dense.items():
        have = set(entry["fields"])
        if "trellis" not in have or not {"suh", "svh"} <= have or len(have & set(MARKER_FIELDS)) != 1:
            problems.append(f"{module}: incomplete field set {sorted(have)}")
    if unknown_fields:
        problems.append(f"unknown fields: {unknown_fields[:5]}")
    for module in sorted(vision_dups):
        fused = module.rsplit(".", 1)[0] + ".qkv.weight"
        if fused not in weight_map:
            problems.append(f"{module}: trellis q/k/v without BF16 {fused}")
    if any(k.startswith("mcg") for k in markers) and any(k.startswith("mul1") for k in markers):
        problems.append("artifact mixes mcg and mul1 codebooks")

    codebook = "mul1" if fields["mul1"] else "mcg" if fields["mcg"] else None
    all_expert_bits = collections.Counter()
    for c in expert_bits.values():
        all_expert_bits.update(c)
    moe_reasons = []
    if codebook not in SPARKINFER_CODEBOOKS:
        moe_reasons.append(f"codebook {codebook!r} not decodable by sparkinfer (MCG-only decoder)")
    if not index_only and not set(all_expert_bits) <= SPARKINFER_TRELLIS_BITS:
        moe_reasons.append(f"expert trellis bits {sorted(all_expert_bits)} outside sparkinfer's {sorted(SPARKINFER_TRELLIS_BITS)}")
    dense_native = {}
    for module, entry in dense.items():
        ok = codebook in SPARKINFER_CODEBOOKS and entry.get("bits") in SPARKINFER_TRELLIS_BITS and entry.get("dims_div_128", False)
        dense_native[module] = bool(ok) and not index_only

    receipt = {
        "schema": SCHEMA,
        "index_path": str(index_path),
        "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "index_only": index_only,
        "total_size": index.get("metadata", {}).get("total_size"),
        "tensors": len(weight_map),
        "shards": shards,
        "patterns": dict(sorted(patterns.items())),
        "fields": dict(fields),
        "codebook": codebook,
        "markers": dict(markers),
        "experts": {
            "layers": sorted(expert_layers),
            "nextn_layer": NEXTN_LAYER,
            "bits_histogram": {str(k): v for k, v in sorted(all_expert_bits.items())},
            "layer_bits": {str(l): {str(k): v for k, v in sorted(c.items())} for l, c in sorted(expert_bits.items())},
        },
        "dense": {m: {k: v for k, v in e.items() if k != "fields"} | {"fields": sorted(set(e["fields"]))} for m, e in sorted(dense.items())},
        "vision_qkv_duplicates": sorted(vision_dups),
        "problems": problems,
        "verdict": {
            "contract_ok": not problems,
            "moe_sparkinfer_native": not moe_reasons and not index_only,
            "moe_reasons": moe_reasons + (["index-only census: shard headers not read"] if index_only else []),
            "dense_sparkinfer_native": dense_native,
        },
    }
    return receipt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="model directory (or the index file with --index-only)")
    ap.add_argument("--index-only", action="store_true", help="names only; no shard headers, no markers")
    ap.add_argument("--out", type=Path, help="write the JSON receipt here")
    args = ap.parse_args(argv)
    path = Path(args.path)
    index_path = path if path.is_file() else path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise SystemExit(f"census: no index at {index_path}")
    receipt = census(index_path, index_only=args.index_only)
    text = json.dumps(receipt, indent=1)
    if args.out:
        args.out.write_text(text)
    v = receipt["verdict"]
    dense_ok = sum(v["dense_sparkinfer_native"].values())
    print(f"census: {receipt['tensors']} tensors, {len(receipt['shards'])} shards, codebook={receipt['codebook']}, "
          f"expert layers={receipt['experts']['layers'][0]}..{receipt['experts']['layers'][-1]} "
          f"({len(receipt['experts']['layers'])}), expert bits={receipt['experts']['bits_histogram'] or 'n/a'}")
    print(f"census: dense modules={len(receipt['dense'])} native-capable={dense_ok}; "
          f"moe native={v['moe_sparkinfer_native']} {v['moe_reasons']}")
    for p in receipt["problems"]:
        print(f"census: PROBLEM {p}", file=sys.stderr)
    return 0 if v["contract_ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
