"""Verify byte-for-byte preservation of the actual handoff's recovered evidence."""
import argparse
import json
from pathlib import Path

from medtoolbench.research.build import load_gzip
from medtoolbench.research.job_state import write_json

p = argparse.ArgumentParser()
p.add_argument("--deployment", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
if a.out.exists():
    raise RuntimeError("Recovery verification output exists")
handoff = json.loads((a.deployment.parent / "handoff.json").read_text())
reports = []
for name, proof in handoff["partial_evidence_audits"].items():
    source = Path(proof["source"])
    target = source.parent / ("attempt" + str(int(source.name[7:]) + 1))
    recovered = json.loads((target / "recovery.json").read_text())
    if Path(recovered["source"]).resolve() != source.resolve():
        raise RuntimeError("Recovery source mismatch")
    keys = recovered["recovered_episode_keys"]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate recovered episode keys")
    states = set()
    for key in keys:
        relative = Path("episodes") / (key + ".json.gz")
        if (source / relative).read_bytes() != (target / relative).read_bytes():
            raise RuntimeError("Recovered episode bytes changed")
        states.add(load_gzip(source / relative)["before_ref"])
    for ref in states:
        if (source / ref).read_bytes() != (target / ref).read_bytes():
            raise RuntimeError("Recovered initial state bytes changed")
    reports.append({"job": name, "recovered_episodes": len(keys), "initial_states": len(states),
                    "source": str(source), "target": str(target), "byte_identical": True})
result = {"all_pass": True, "jobs": reports, "recovered_episodes": sum(r["recovered_episodes"] for r in reports),
          "scope": "Byte preservation of already-audited source evidence; not additional benchmark episodes"}
write_json(a.out, result)
print(json.dumps(result), flush=True)
