# MedToolFault and MedGuard

**MedToolFault** evaluates clinical tool-using agents under controlled tool faults. **MedGuard** adds read verification and ambiguous-write attribution through the public tool interface.

This repository contains the study implementation, deterministic benchmark-construction code, frozen primary-task and external-question selection indices, and aggregate Source Data. The internal Python package remains `medtoolbench` so that historical experiment identifiers stay stable.

## Release contents

- `repo/medtoolbench/research/`: source-task construction, state contracts, fault injection, MedGuard, local-model execution and audit/statistical analysis.
- `repo/medtoolbench/supplement/`: mechanism controls, task expansion, call-budget tests, external original-answer evaluation and model adapters.
- `repo/medtoolbench/publication/`: frozen-evidence resolution and post-hoc analysis.
- `repo/tests/`: regression tests; `repo/scripts/`: study orchestration utilities.
- `benchmark/primary_index.json`: 300 primary tasks, comprising 45 development and 255 held-out test tasks, with source IDs, splits, eligible faults and fixture hashes.
- `benchmark/external_selection_index.json`: the final 100-question, 45-patient-group external test selection.
- `results/source_data/`: the nine numerical exports used in the manuscript analyses, including 348 paired effects and 1,991 endpoint rows.
- `RELEASE_MANIFEST.json`: checksums and the explicit release scope.

The main study contains 127,861 formal episode evaluations; these are repeated evaluations, not independent patients. The primary study and follow-ups have different model cohorts and protocols. Consult the source-data settings and individual runners before reproducing a particular arm.

## CPU-only quick start

Use Python 3.10 or newer. From the repository root:

```bash
python -m pip install -e repo
python -m medtoolbench.build --out runs/synthetic_smoke --tasks 8
python -m unittest discover -s repo/tests -p test_core.py -v
```

The smoke command generates programmatic synthetic fixtures and tests their scripted recovery oracle. It does **not** reproduce the paper's clinical-source experiment and is not included in its formal episode total. Use a new output directory for each run.

For the full CPU regression suite and figure-design checks, install NumPy and Matplotlib, then run:

```bash
python -m unittest discover -s repo/tests -v
```

Figure tests require an installed Times-family font, such as Nimbus Roman. Checks requiring a pinned GPT-OSS tokenizer or full historical campaign files explicitly skip when these separately obtained fixtures are absent; a portable pass does not certify those fixture-dependent checks. Model inference is local; model weights and GPU-dependent runtime packages are obtained separately. `repo/requirements-release.in` records one study environment and is not a universal environment specification for every adapter.

## Clinical-source benchmark construction

Read [benchmark/README.md](benchmark/README.md) for source revisions, reconstruction commands and selection metadata. Raw FHIR resources, containers, model weights and full episode traces are not bundled. Obtain upstream resources from their original providers and use the supplied constructors and checksums.

The provided benchmark indices specify the evaluated selection; they are not standalone patient fixtures. Reproducing all study results also requires the original source resources, arm-specific protocols and runtime environments. This initial release does not claim a one-command reconstruction of every historical campaign.

## Interpretation

The benchmark separates answer correctness, required record changes and forbidden mutations. MedGuard improves conditional recovery in the primary study and eliminates duplicates in its tested committed-timeout cases. Expanded tasks, delayed visibility and concurrent writes expose limitations, and the external informed-baseline comparisons remain inconclusive. Source Data retain positive and adverse results.

The implementation is a research evaluation framework for isolated environments, not clinical decision or deployment software. Study-specific task contracts define the reported endpoints.

## Sources and reuse

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream attribution and source terms. A project-wide license for the authors' original contributions has not yet been designated; no new license grant is implied by this initial upload. Upstream materials retain their own terms.
