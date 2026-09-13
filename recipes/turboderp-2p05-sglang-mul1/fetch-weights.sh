#!/usr/bin/env bash
# Pinned, SHA-verified download of turboderp/GLM-5.3-Flash-exl3 @ 2.05bpw.
set -u
REV=51058cd551c7e570d87bd32a4adee720edce2349
DL="https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/resolve/$REV"
BASE="${1:-./model}"
mkdir -p "$BASE" && cd "$BASE" || exit 1
python3 - "$DL" <<'PY'
import json, subprocess, sys, urllib.request, hashlib, os
dl = sys.argv[1]
req = urllib.request.Request(f"https://huggingface.co/api/models/turboderp/GLM-5.3-Flash-exl3/revision/2.05bpw?blobs=true")
info = json.load(urllib.request.urlopen(req))
assert info["sha"] == "51058cd551c7e570d87bd32a4adee720edce2349", "branch head moved; pin broken"
rows = [(s["rfilename"], int(s.get("size") or 0), (s.get("lfs") or {}).get("sha256") or "")
        for s in info["siblings"] if not s["rfilename"].endswith(".kate-swp")]
open("files.tsv", "w").write("\n".join("\t".join(map(str, r)) for r in rows) + "\n")
PY
while IFS=$'\t' read -r path size sha; do
  [ -n "$path" ] || continue
  [ -f "$path" ] && [ "$(stat -c%s "$path" 2>/dev/null)" = "$size" ] && continue
  curl -sSL -C - --fail -o "$path.part" "$DL/$path" || { echo "download failed: $path" >&2; exit 2; }
  [ "$(stat -c%s "$path.part")" = "$size" ] || { echo "size mismatch: $path" >&2; exit 3; }
  if [ -n "$sha" ]; then
    [ "$(sha256sum "$path.part" | cut -d' ' -f1)" = "$sha" ] || { echo "sha mismatch: $path" >&2; exit 4; }
  fi
  mv -f "$path.part" "$path" && echo "verified $path"
done < files.tsv
rm -f files.tsv
echo "DONE — model pinned at $REV in $BASE"
