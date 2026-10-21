"""Verify that git lfs pull restored the actual runtime files."""
import hashlib
import json
from pathlib import Path


root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "runtime_assets.json").read_text(encoding="utf-8"))
for asset in manifest["files"]:
    path = root / asset["path"]
    assert path.is_file(), f"Missing {asset['path']}; run git lfs pull"
    assert path.stat().st_size == asset["bytes"], f"Incorrect size: {asset['path']}; run git lfs pull"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    assert digest.hexdigest() == asset["sha256"], f"Incorrect SHA256: {asset['path']}"
print(f"Verified {len(manifest['files'])} runtime files.")
