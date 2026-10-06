"""Read-only export from an isolated copy of the author's original HAPI DB."""
import argparse
import json
from pathlib import Path

from .fhir import HapiProcess
from medtoolbench.storage import digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if (args.out / "complete.json").exists():
        raise SystemExit("Export is already complete")
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.upstream / "MedAgentBench/data/medagentbench/test_data_v2.json"
    tasks = json.loads(source.read_text())
    (args.out / "source_tasks.json").write_text(json.dumps(tasks, indent=2) + "\n")
    patients = sorted({t["eval_MRN"] for t in tasks if t.get("eval_MRN")})
    with HapiProcess(args.upstream / "medagentbench.sif", args.out / "hapi",
                     args.upstream / "fhir_root/data/test_db.mv.db", heap="3g") as server:
        # Include all patient resources for real patient-lookup negatives and distractors.
        plist = server.all_pages("/Patient?_count=2000")
        (args.out / "patients.json").write_text(json.dumps(plist) + "\n")
        counts = []
        for patient in patients:
            target = args.out / (patient + ".json")
            if target.exists():
                rows = json.loads(target.read_text())
            else:
                rows = []
                for kind in ("Observation", "MedicationRequest", "ServiceRequest"):
                    rows.extend(server.all_pages(f"/{kind}?subject=Patient/{patient}&_count=2000"))
                target.write_text(json.dumps(rows) + "\n")
            counts.append({"patient": patient, "resources": len(rows), "sha256": digest(rows)})
            print(json.dumps(counts[-1]), flush=True)
    report = {"source": str(source), "tasks": len(tasks), "task_hash": digest(tasks),
              "patient_resources": len(plist), "task_patients": len(patients), "counts": counts,
              "fhir_mutations": 0, "note": "Raw source export, not a corrected benchmark or redistributed data release."}
    (args.out / "complete.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
