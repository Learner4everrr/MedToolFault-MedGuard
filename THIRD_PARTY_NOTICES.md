# Upstream attribution and release scope

MedToolFault builds on public clinical-agent task environments and data resources. The original authors and sources retain their respective rights and terms.

## MedAgentBench

Jiang, Yixing; Black, Kameron C.; Geng, Gloria; Park, Danny; Zou, James; Ng, Andrew Y.; Chen, Jonathan H. **MedAgentBench: A Virtual EHR Environment to Benchmark Medical LLM Agents.** NEJM AI (2025), AIdbp2500144.

Source: https://github.com/stanfordmlgroup/MedAgentBench

The upstream repository supplies an MIT license, copied without modification in `third_party/MedAgentBench-LICENSE.txt`. Our constructors deduplicate source tasks, assign patient-group splits, adapt serialization, derive explicit contracts and introduce controlled tool faults. The original container and FHIR database are obtained from the upstream provider rather than redistributed here.

## FHIR-AgentBench

Source: https://github.com/glee4810/FHIR-AgentBench

The upstream repository supplies Creative Commons Attribution 4.0 International, copied without modification in `third_party/FHIR-AgentBench-LICENSE.txt`. The external evaluation retains original answers and introduces local tools, fault conditions, bounded execution and disclosed serialization adapters. The release includes the selected question identifiers and aggregate outcomes, not the original clinical records or full answer dataset.

## MIMIC-IV Clinical Database Demo on FHIR

Source: https://physionet.org/content/mimic-iv-fhir-demo/2.1.0/

The local source distribution includes the Open Database License. This release does not mirror that database. Obtain the original demo records from PhysioNet and follow the provider's current terms. Model checkpoints and HAPI container images are also obtained separately under their source terms.

## Project contributions

The code, task construction, explicit contracts, verification wrapper, experimental orchestration and result exports are provided for inspection and reproduction. A project-wide reuse license for the authors' original contributions has not yet been selected. This notice preserves upstream attribution; it does not relicense third-party material or imply upstream endorsement.
