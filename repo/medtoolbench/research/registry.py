"""Resolve and verify planned local model snapshots without copying weights."""
import argparse
import json
from pathlib import Path


CANDIDATES = [
    ("qwen35_9b", "Qwen/Qwen3.5-9B", "Qwen", False, 4, "Qwen3.5-9B"),
    ("ministral3_8b", "mistralai/Ministral-3-8B-Instruct-2512", "Mistral", False, 4, "Ministral-3-8B-Instruct-2512"),
    ("qwen25_7b", "Qwen/Qwen2.5-7B-Instruct", "Qwen", False, 4, None),
    ("qwen25_14b", "Qwen/Qwen2.5-14B-Instruct", "Qwen", False, 2, None),
    ("qwen3_8b", "Qwen/Qwen3-8B", "Qwen", False, 4, None),
    ("qwen3_8b_thinking", "Qwen/Qwen3-8B", "Qwen", True, 4, None),
    ("llama31_8b", "meta-llama/Llama-3.1-8B-Instruct", "Llama", False, 4, None),
    ("gemma3_12b", "google/gemma-3-12b-it", "Gemma", False, 2, None),
    ("phi4_mini", "microsoft/Phi-4-mini-instruct", "Phi", False, 4, None),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--new-models", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit("Do not overwrite a model registry")
    records = []
    for name, repo_id, family, thinking, batch, local in CANDIDATES:
        snapshots = [a.new_models / local] if local else sorted((a.cache / ("models--" + repo_id.replace("/", "--")) / "snapshots").glob("*"))
        if len(snapshots) != 1:
            raise ValueError(f"Ambiguous or absent cached snapshot for {repo_id}: {snapshots}")
        path = snapshots[0].resolve()
        cfg = json.loads((path / "config.json").read_text())
        index = path / "model.safetensors.index.json"
        names = set(json.loads(index.read_text())["weight_map"].values()) if index.exists() else {"model.safetensors"}
        missing = [f for f in names if not (path / f).is_file()]
        if missing:
            raise ValueError(f"Missing local weights for {repo_id}: {missing}")
        download = path / "DOWNLOAD_MANIFEST.json"
        revision = json.loads(download.read_text())["revision"] if download.exists() else path.name
        records.append({"name": name, "repo_id": repo_id, "family": family, "thinking": thinking,
                        "path": str(path), "revision": revision, "batch_size": batch,
                        "model_type": cfg["model_type"], "weight_bytes": sum((path / f).stat().st_size for f in names),
                        "preexisting_shared_cache": local is None, "inference_api": False,
                        "status": "candidate; require development adapter validation before held-out execution"})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"systems": records, "independent_families": len({r["family"] for r in records}),
        "distinct_weight_snapshots": len({r["path"] for r in records}), "new_weight_copies_created_by_registry": 0}, indent=2) + "\n")
    print(json.dumps([{k: r[k] for k in ("name", "family", "revision", "weight_bytes", "preexisting_shared_cache")} for r in records], indent=2))


if __name__ == "__main__":
    main()
