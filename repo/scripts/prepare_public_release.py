"""Export an allowlisted public release; never copy raw clinical records or model weights."""
import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label(value):
    return hashlib.sha256(str(value).encode()).hexdigest()[:20]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.workspace.resolve(), args.out.resolve()
    if root == out or out == root / "repo":
        raise SystemExit("Use a separate release checkout")
    if (out / "RELEASE_MANIFEST.json").exists():
        raise SystemExit("Release already exported; do not overwrite it")
    copied = {}

    def copy(source, relative):
        target = out / relative
        if target.exists():
            raise SystemExit("Refusing to overwrite " + str(relative))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied[str(relative)] = sha(source)

    for folder in ("medtoolbench", "scripts", "tests"):
        for source in sorted((root / "repo" / folder).rglob("*.py")):
            if source.relative_to(root).as_posix() == "repo/medtoolbench/research/environment_check.py":
                continue  # Optional host-specific package check, not a benchmark dependency.
            if source.is_symlink() or "__pycache__" in source.parts:
                raise SystemExit("Unexpected source link/cache")
            copy(source, source.relative_to(root))
    for name in ("pyproject.toml", "requirements-release.in", "paper_nc_v1/figure_design.py"):
        copy(root / "repo" / name, Path("repo") / name)
    for source in sorted((root / "repo/paper_nc_v1/source_data").glob("*.csv")):
        copy(source, Path("results/source_data") / source.name)
    for project in ("MedAgentBench", "FHIR-AgentBench"):
        copy(root / "data/upstream" / project / "LICENSE", Path("third_party") / (project + "-LICENSE.txt"))

    tasks_path = root / "data/authentic_v2/tasks.json"
    tasks = json.loads(tasks_path.read_text())
    rows = [{"id": t["id"], "template": t["template"], "split": t["split"],
             "group_id": label(t["group"]), "hazards": t["hazards"],
             "action_required": bool(t["contract"]["required_creations"]),
             "source_id": t["source"]["id"], "source_version": t["source"]["version"],
             "source_identity": t["source"]["identity"], "fixture_hash": t["fixture_hash"]}
            for t in tasks]
    splits = dict(Counter(t["split"] for t in tasks))
    if len(tasks) != 300 or splits != {"test": 255, "dev": 45}:
        raise SystemExit("Unexpected primary benchmark inventory")
    write_json(out / "benchmark/primary_index.json", {
        "description": "Frozen task index; source prompts, reference answers and FHIR records are obtained upstream.",
        "full_local_manifest_sha256": sha(tasks_path), "task_count": 300,
        "splits": splits, "groups": {s: len({t["group"] for t in tasks if t["split"] == s}) for s in splits},
        "tasks": rows})
    selection_path = root / "data/b_lite_v1/selection.json"
    selection = json.loads(selection_path.read_text())
    selected = [{"id": t["id"], "group_id": label(t["patient"]),
                 "template_id": label(t["template"])} for t in selection["selected"]]
    if len(selected) != 100 or len({t["group_id"] for t in selected}) != 45:
        raise SystemExit("Unexpected external selection")
    write_json(out / "benchmark/external_selection_index.json", {
        "description": "Final B-lite test selection; original questions, answers and data are obtained upstream.",
        "seed": selection["seed"], "source_selection_sha256": sha(selection_path),
        "questions": selected, "patient_groups": 45,
        "templates": len({t["template_id"] for t in selected})})
    plan = json.loads((root / "data/primary_study_v1/plan.json").read_text())
    write_json(out / "benchmark/primary_provenance.json", {
        "frozen_at": plan["frozen_at"], "manifest_hash": plan["manifest_hash"],
        "core_hashes": plan["core_hashes"], "image_sha256": plan["image_sha256"],
        "note": "Frozen hashes describe the historical experiment; current release source hashes are recorded separately.",
        "medagentbench_revision": "99260117137b09f04837a8c18d18a1107efa55ae",
        "fhir_agentbench_revision": "bbb42909a5a7eb907d1cd91f72a560729e7037ea"})
    patterns = [r"gh[pousr]_[A-Za-z0-9]{30,}", r"github_pat_[A-Za-z0-9_]{40,}",
                r"hf_[A-Za-z0-9]{25,}", r"sk-[A-Za-z0-9_-]{25,}",
                r"AKIA[A-Z0-9]{16}", r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"]
    files = [p for p in out.rglob("*") if p.is_file() and ".git" not in p.relative_to(out).parts]
    for path in files:
        if path.stat().st_size > 10_000_000:
            raise SystemExit("Unexpected large release file: " + str(path.relative_to(out)))
        text = path.read_text()
        if any(re.search(pattern, text) for pattern in patterns):
            raise SystemExit("Potential credential in " + str(path.relative_to(out)))
    write_json(out / "RELEASE_MANIFEST.json", {
        "version": "2026-10-06-initial-code-and-benchmark-index",
        "copied_source_sha256": copied,
        "file_sha256": {str(p.relative_to(out)): sha(p) for p in sorted(files)},
        "raw_clinical_records_included": False, "model_weights_included": False,
        "private_author_checklist_included": False,
        "scope": "Code, benchmark construction and selection metadata, aggregate Source Data; upstream data obtained separately."})
    print(json.dumps({"files": len(files) + 1, "primary_tasks": len(rows),
                      "external_questions": len(selected), "credential_pattern_scan": "pass"}, indent=2))


if __name__ == "__main__":
    main()
