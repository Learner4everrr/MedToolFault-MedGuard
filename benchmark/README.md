# Benchmark construction and frozen selection

MedToolFault links six tool-failure classes to patient-specific answer and record-state contracts:

| Failure class | Implemented assignments | MedGuard operation |
|---|---|---|
| Interface drift | `interface_drift` | Metadata-guided request repair |
| Transient service error | `transient_503` | Bounded retry |
| Stale or incomplete evidence | `stale`; external partial-history variants | Authoritative re-read and pagination |
| Patient mismatch | `patient_mismatch` | Patient-reference verification |
| Provenance conflict | `provenance_conflict` | Finalized, source-of-record evidence checks |
| Ambiguous write | `commit_timeout`, `no_commit_timeout` | Compare equivalent records before and after the write |

The last class contains two assignments with the same visible timeout but different underlying commit states. `clean` is the reference condition. Eligibility is task-specific; consult `primary_index.json` rather than assigning every fault to every task.

## Frozen sources

- [MedAgentBench](https://github.com/stanfordmlgroup/MedAgentBench), revision `99260117137b09f04837a8c18d18a1107efa55ae`.
- [FHIR-AgentBench](https://github.com/glee4810/FHIR-AgentBench), revision `bbb42909a5a7eb907d1cd91f72a560729e7037ea`.
- [MIMIC-IV Clinical Database Demo on FHIR 2.1.0](https://physionet.org/content/mimic-iv-fhir-demo/2.1.0/).

Download source records and containers from the original providers. The primary export expects this local layout:

```text
data/upstream/
  MedAgentBench/                 # pinned upstream checkout
  medagentbench.sif              # Apptainer/Singularity image
  fhir_root/data/test_db.mv.db   # original source HAPI database
```

After acquiring those resources, use a new output directory for each construction or validation step:

```bash
python -m medtoolbench.research.export_mab --upstream data/upstream --out data/mab_export
python -m medtoolbench.research.build --export data/mab_export --upstream data/upstream --out data/authentic
python -m medtoolbench.research.validate --benchmark data/authentic --image data/upstream/medagentbench.sif --out data/oracle_validation --split all --workers 1
```

These commands require the upstream HAPI image and an Apptainer/Singularity runtime. They perform source export, deterministic construction and real-HAPI scripted validation; they do not start model inference. A mutable upstream image tag may produce different content from the historical source image. `primary_provenance.json` records the historical image checksum and semantic task-manifest hash for comparison, not a guarantee that rebuilding a SIF is byte-identical across runtimes.

`primary_index.json` contains source IDs, template IDs, development/test membership, opaque group identifiers, eligible hazards and fixture hashes for all 300 constructed tasks. Clinical values, names, original prompts and reference answers are omitted from this index and are reconstructed from source data. Group identifiers here are deterministic hashes of the local source-group identifiers, not new patient observations.

The test partition contains 255 tasks from 88 groups, with 255 clean and 806 eligible fault assignments per policy. The development partition contains 45 tasks from 12 groups. Primary construction and validation remain separate steps.

The external test index preserves the final 100 selected question IDs and opaque group/template identifiers. The selection derives from a previously unattempted remainder of the original external campaign, not a simple random sample of the full upstream benchmark. Original questions, answers and clinical records are obtained upstream. External preparation and final protocol logic are in `repo/medtoolbench/research/external_inventory.py` and `repo/medtoolbench/supplement/b_lite.py`; model-specific adapters are in the same package.

## Complete state-based evaluation

The full contracts and graders are implemented in `repo/medtoolbench/research/contracts.py`, and the primary proxy/guard in `repo/medtoolbench/research/transport.py`. Task success requires the requested answer and state transition together with the absence of forbidden mutation. An omitted action can fail a task without constituting a forbidden mutation; the two endpoints are kept separate.

The synthetic quick-start fixture uses `repo/medtoolbench/construction.py` and a SQLite store for a small CPU demonstration. The formal clinical-source study uses real HAPI and distinct construction code. Results from the demonstration must not be substituted for the paper's formal results.
