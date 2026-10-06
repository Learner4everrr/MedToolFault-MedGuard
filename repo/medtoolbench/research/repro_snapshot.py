"""Archive source and environment metadata without copying models or clinical data."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit("Snapshot already exists; choose a new version")
    a.out.mkdir(parents=True)
    packages = sorted([{"name": d.metadata["Name"], "version": d.version,
                        "location": str(d.locate_file("")), "requires": d.requires or []}
                       for d in importlib.metadata.distributions()], key=lambda d: (d["name"].lower(), d["location"]))
    manifest = {}
    with zipfile.ZipFile(a.out / "source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(a.repo.rglob("*")):
            if not path.is_file() or path.is_symlink() or any(s.startswith(".") or s == "__pycache__" for s in path.relative_to(a.repo).parts):
                continue
            if path.suffix not in {".py", ".md", ".toml", ".yaml", ".yml", ".json", ".sh", ".txt", ".in", ".lock"}:
                continue
            data = path.read_bytes()
            name = str(path.relative_to(a.repo))
            manifest[name] = hashlib.sha256(data).hexdigest()
            archive.writestr(name, data)
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "executable": sys.executable,
              "python": sys.version, "platform": platform.platform(), "packages": packages,
              "source_sha256": manifest, "contains_model_weights": False, "contains_clinical_records": False,
              "status": "Reproducibility snapshot, not a dependency-resolved portable environment lock"}
    (a.out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"source_files": len(manifest), "package_entries": len(packages), "out": str(a.out)}, indent=2))


if __name__ == "__main__":
    main()
