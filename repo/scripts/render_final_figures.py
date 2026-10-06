"""English, evidence-derived presentation figures; no inference or regrading.

Run from project root:
env PYTHONPATH=repo MPLCONFIGDIR=/tmp/medtoolbench-figure-cache env/bin/python \
  repo/scripts/render_final_figures.py --out repo/reports/2026-10-01/english_figures_v1
"""
import argparse
import csv
import hashlib
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from medtoolbench.publication.evidence import Evidence, LABELS, MODELS, effect_rows, require


COLORS = {"vanilla": "#6B7280", "informed": "#D59224", "medguard": "#007D88"}
TAXONOMY = [
    ("Interface drift", "A search parameter changes; a previously valid request is rejected.",
     "Inspect public metadata and retry with the documented interface."),
    ("Transient failure", "A temporary service error interrupts retrieval or execution.",
     "Use bounded recovery; do not treat a write timeout as proof that nothing was committed."),
    ("Stale or incomplete data", "An old observation or partial history is returned as if it were sufficient.",
     "Re-read the authoritative source; recover pages and apply the requested time/order constraints."),
    ("Patient mismatch", "Returned observations belong to a different patient.",
     "Check patient references and re-read the requested patient's authoritative records."),
    ("Provenance conflict", "A preliminary or non-authoritative result conflicts with the source of record.",
     "Verify source and finalized status; use authoritative finalized observations."),
    ("Ambiguous write outcome", "A timeout occurs after commit, or before commit; the response alone cannot distinguish them.",
     "Compare equivalent records before and after the write; recover a committed result or retry only when justified."),
]


def csv_file(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save(fig, out, stem, bundle):
    fig.savefig(out / (stem + ".png"), dpi=220, facecolor="white")
    fig.savefig(out / (stem + ".svg"), metadata={"Date": None}, facecolor="white")
    fig.savefig(out / (stem + ".pdf"), metadata={"CreationDate": None, "ModDate": None}, facecolor="white")
    bundle.savefig(fig, facecolor="white")
    plt.close(fig)


def title(fig, heading, subtitle):
    fig.text(.04, .955, heading, fontsize=22, weight="bold", color="#153344", va="top")
    fig.text(.04, .900, subtitle, fontsize=11.5, color="#485968", va="top")


def tidy(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#C5CED4")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#E7ECEF", linewidth=.7)
    ax.set_axisbelow(True)


def taxonomy(out, bundle):
    fig, ax = plt.subplots(figsize=(15, 8.4))
    ax.axis("off")
    title(fig, "Six tool-failure classes and MedGuard recovery",
          "Clinical-tool reliability: verifying the patient, evidence and persistent state, not retraining the model")
    rows = [[textwrap.fill(a, 20), textwrap.fill(b, 43), textwrap.fill(c, 64)] for a, b, c in TAXONOMY]
    table = ax.table(cellText=rows, colLabels=["Failure class", "What can go wrong", "MedGuard recovery mechanism"],
                     colWidths=[.19, .33, .48], bbox=[0, 0, 1, 1], cellLoc="left", colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#D9E2E7")
        cell.PAD = .06
        if r == 0:
            cell.set_facecolor("#174C60")
            cell.set_text_props(color="white", weight="bold")
            cell.set_height(.08)
        else:
            cell.set_height(.145)
            cell.set_facecolor("#EFF6F7" if r % 2 else "white")
            cell.set_text_props(color="#213A48", weight="bold" if c == 0 else "normal")
    fig.subplots_adjust(left=.04, right=.96, top=.83, bottom=.16)
    fig.text(.04, .105, "Scope: public recovery interfaces; no hidden answers. B-lite tests only the five read-side classes.", fontsize=11)
    fig.text(.04, .065, "Boundary: write attribution assumes a single writer and immediately visible commits; delayed visibility and concurrency remain limitations.", fontsize=10.4, color="#A64532")
    save(fig, out, "01_failure_classes_and_recovery", bundle)
    csv_file(out / "01_failure_classes_and_recovery.csv", [dict(failure=a, example=b, recovery=c) for a, b, c in TAXONOMY])


def primary(data, out, bundle):
    models = list(data["primary"]["models"])
    fig, ax = plt.subplots(figsize=(13, 8.4))
    title(fig, "Primary study: MedGuard improves recovery under tool faults",
          "Paired safe-success difference versus vanilla; 95% patient-cluster bootstrap intervals")
    for i, model in enumerate(models):
        result = data["primary"]["models"][model]
        for field, offset, color, label in (
                ("paired_guard_gain", -.13, "#6B7280", "All assigned test tasks (255 per system)"),
                ("paired_guard_gain_clean_capable", .13, "#007D88", "Vanilla-clean-capable subset (primary endpoint)")):
            e = result[field]
            point = 100 * e["task_weighted_gain"]
            lo, hi = [100 * v for v in e["patient_cluster_ci95"]]
            y = len(models) - 1 - i + offset
            ax.plot([lo, hi], [y, y], color=color, linewidth=2.4, solid_capstyle="round")
            ax.scatter([point], [y], color=color, s=42, label=label if i == 0 else None, zorder=3)
        ax.text(59, len(models) - 1 - i, "n = " + str(result["clean_capable_baseline_tasks"]),
                va="center", fontsize=10.5, color="#007D88")
    ax.axvline(0, color="#AE5145", linestyle="--", linewidth=1)
    ax.set_yticks(range(len(models)), [LABELS[m] for m in reversed(models)])
    ax.set_xlim(-6, 67)
    ax.set_ylim(-.7, len(models) - .3)
    ax.set_xticks(range(0, 61, 10))
    ax.set_xlabel("MedGuard − vanilla safe success (percentage points)", labelpad=12)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.11), frameon=False, fontsize=10)
    tidy(ax)
    fig.subplots_adjust(left=.27, right=.96, top=.77, bottom=.17)
    fig.text(.04, .087, "Nine configurations / eight checkpoints; 47,745 episodes, 255 test tasks and 88 patient groups. n denotes clean-capable tasks.", fontsize=10.3)
    fig.text(.04, .048, "Safe success requires both the correct answer and permitted final state. Conditional effects are not whole-population benefits.", fontsize=10.3)
    save(fig, out, "02_primary_recovery_effects", bundle)


def grouped_panel(ax, data, metric, denominator, heading, unsafe=False):
    offsets = {"vanilla": -.24, "informed": 0, "medguard": .24}
    for condition in offsets:
        positions = np.arange(4)[::-1] + offsets[condition]
        counts = [data[m]["conditions"][condition][metric]["count"] for m in MODELS]
        values = [100 * n / denominator for n in counts]
        ax.barh(positions, values, height=.205, color=COLORS[condition], label=condition)
        for y, value in zip(positions, values):
            ax.text(value + 1.0, y, f"{value:.1f}", va="center", fontsize=8.6)
    ax.set_yticks(np.arange(4)[::-1], [LABELS[m] for m in MODELS])
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_ylim(-.6, 3.6)
    ax.set_title(heading, loc="left", fontsize=12, weight="bold", pad=15)
    ax.set_xlabel(("Unsafe final state (%) · lower is better" if unsafe else "Safe success (%) · higher is better"), fontsize=10)
    tidy(ax)


def controls(data, out, bundle):
    a = {m: data["a"]["reports"]["matched_controls/" + m] for m in MODELS}
    d = {m: data["d"]["reports"]["strict_raw_budget_12/" + m] for m in MODELS}
    fig, axs = plt.subplots(1, 3, figsize=(18, 7))
    title(fig, "Controls, equal budgets and safety trade-offs",
          "Informed agents receive explicit verification/recovery instructions; MedGuard executes recovery at the tool layer")
    grouped_panel(axs[0], a, "fault_success", 806, "A  Matched controls\n806 fault assignments per policy/model")
    grouped_panel(axs[1], d, "fault_success", 160, "D  Same 12-call ceiling\n160 fault assignments per policy/model")
    grouped_panel(axs[2], a, "action_required_unsafe_state", 202,
                  "A  Action-required safety\n202 fault assignments per policy/model", unsafe=True)
    axs[1].set_yticklabels([])
    axs[2].set_yticklabels([])
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, ["Vanilla", "Informed", "MedGuard"], loc="upper right", bbox_to_anchor=(.97, .86), ncol=3, frameon=False)
    fig.subplots_adjust(left=.12, right=.97, top=.73, bottom=.20, wspace=.16)
    fig.text(.04, .107, "Raw proportions, not paired effect estimates. The shared hard ceiling does not imply equal realized calls, tokens or runtime.", fontsize=11)
    fig.text(.04, .060, "Safety counterexample: Llama unsafe action states increase from 106/202 (vanilla) to 172/202 (MedGuard). These are contract violations, not patient-harm rates.", fontsize=10.6, color="#A64532")
    save(fig, out, "03_controls_budget_and_safety", bundle)


def external(data, out, bundle):
    fig, axs = plt.subplots(1, 2, figsize=(15, 7.2))
    title(fig, "External B-lite: gains are model- and comparator-dependent",
          "100 original questions, 45 patients, 25 templates; 7,200 complete read-only retrieval/calculation episodes")
    left, right = axs
    for j, condition in enumerate(("vanilla", "informed", "medguard")):
        ys = np.arange(4)[::-1] + (j - 1) * .24
        vals = [data["b"][m]["conditions"][condition]["correct_fault"] / 5 for m in MODELS]
        left.barh(ys, vals, height=.205, color=COLORS[condition], label=condition)
        for y, v in zip(ys, vals):
            left.text(v + .5, y, f"{v:.1f}%", fontsize=9.5, va="center")
    left.set_yticks(np.arange(4)[::-1], [LABELS[m] for m in MODELS])
    left.set_xlim(0, 46)
    left.set_xticks([0, 10, 20, 30, 40])
    left.set_title("A  Fault answer correctness\n500 assignments per model/policy", loc="left", fontsize=12, weight="bold", pad=15)
    left.set_xlabel("Raw answer correctness (%)")
    left.legend(["Vanilla", "Informed", "MedGuard"], frameon=False, ncol=3, fontsize=9.5, loc="upper left", bbox_to_anchor=(0, -.15))
    for i, model in enumerate(MODELS):
        for comp, offset, color, label in (
                ("medguard-vanilla", .12, "#007D88", "MedGuard − vanilla"),
                ("medguard-informed", -.12, "#D59224", "MedGuard − informed")):
            e = data["b"][model]["comparisons"][comp]
            point, lo, hi = [100 * x for x in [e["difference"], *e["ci95"]]]
            y = 3 - i + offset
            right.plot([lo, hi], [y, y], color=color, linewidth=2.5)
            right.scatter([point], [y], color=color, s=45, zorder=3, label=label if i == 0 else None)
    right.axvline(0, color="#A64532", linestyle="--", linewidth=1)
    right.set_yticks(range(4), [])
    right.set_xlim(-9, 18)
    right.set_xticks([-5, 0, 5, 10, 15])
    right.set_title("B  Patient-weighted paired differences\n95% patient-cluster bootstrap intervals", loc="left", fontsize=12, weight="bold", pad=15)
    right.set_xlabel("Answer-correctness improvement (percentage points)")
    right.legend(frameon=False, ncol=2, fontsize=9.5, loc="upper left", bbox_to_anchor=(0, -.15))
    for ax in axs:
        ax.set_ylim(-.55, 3.55)
        tidy(ax)
    fig.subplots_adjust(left=.15, right=.96, top=.72, bottom=.28, wspace=.17)
    fig.text(.04, .093, "All four MedGuard–informed intervals cross zero. The selected unattempted-patient remainder is not a new independently sampled hospital cohort.", fontsize=10.3)
    fig.text(.04, .046, "This read-only task measures answer correctness, not write safety. Model-specific transport adaptations are disclosed; no cross-model prompt-equivalence claim.", fontsize=10.1)
    save(fig, out, "04_external_validation", bundle)


def transfer(data, out, bundle):
    rows = []
    for model in MODELS:
        for arm, label in (("decision_branches", "Decision branches"), ("external_actions", "External-source rule-based actions")):
            result = data["c"]["reports"][arm + "/" + model]
            c = result["conditions"]
            e = result["comparisons"]["medguard-vanilla"]
            rows.append(dict(model=LABELS[model], scenario=label,
                vanilla=100 * c["vanilla"]["fault_success"]["rate"],
                medguard=100 * c["medguard"]["fault_success"]["rate"],
                paired_difference_pp=100 * e["difference"],
                ci_lower_pp=100 * e["ci95"][0], ci_upper_pp=100 * e["ci95"][1]))
    fig, ax = plt.subplots(figsize=(15, 7.8))
    ax.axis("off")
    title(fig, "Task expansion: strong gains coexist with transfer failures",
          "C experiments: semi-synthetic decision/action rules, not authentic treatment-policy evaluation")
    cells = [[r["model"], r["scenario"], f'{r["vanilla"]:.1f}% → {r["medguard"]:.1f}%',
              f'{r["paired_difference_pp"]:+.2f} [{r["ci_lower_pp"]:+.2f}, {r["ci_upper_pp"]:+.2f}]'] for r in rows]
    table = ax.table(cellText=cells, colLabels=["Model", "Scenario", "Raw safe success\nVanilla → MedGuard", "Paired gain, pp\n[95% cluster interval]"],
                     colWidths=[.18, .35, .22, .25], bbox=[0, 0, 1, 1], cellLoc="left", colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(11.5)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#D9E2E7")
        cell.PAD = .06
        if r == 0:
            cell.set_facecolor("#174C60")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#FFF0E9" if r == 2 else "#EFF6F7" if r % 2 else "white")
    fig.subplots_adjust(left=.04, right=.96, top=.82, bottom=.20)
    fig.text(.04, .128, "Boundary test: with delayed write visibility, MedGuard succeeds in 0/60 episodes for each of the four models.", fontsize=11.5, color="#A64532")
    fig.text(.04, .081, "Safety is a separate endpoint: in Ministral decision branches, unsafe action states rise from 172/504 to 307/504 despite higher overall success.", fontsize=10.5)
    fig.text(.04, .043, "Transfer failures remain in the analysis. Trace-level diagnosis is post-hoc; these tables do not establish a single causal explanation.", fontsize=10.5)
    save(fig, out, "05_task_expansion_and_boundaries", bundle)
    csv_file(out / "05_task_expansion_and_boundaries.csv", rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    project, out = args.project.resolve(), args.out.resolve()
    require(out.is_relative_to(project / "repo/reports"), "Use a new directory under repo/reports")
    require(not out.exists(), "Never overwrite existing figure outputs")
    evidence = Evidence(project)
    data = evidence.load()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.labelcolor": "#263D49", "text.color": "#263D49",
                         "svg.fonttype": "none", "pdf.fonttype": 42, "svg.hashsalt": "medtoolbench-final-v1"})
    out.mkdir(parents=True)
    with PdfPages(out / "ALL_FIGURES_EN.pdf", metadata={"Title": "MedToolBench and MedGuard: final experimental figures", "CreationDate": None, "ModDate": None}) as bundle:
        taxonomy(out, bundle)
        primary(data, out, bundle)
        controls(data, out, bundle)
        external(data, out, bundle)
        transfer(data, out, bundle)
    csv_file(out / "all_paired_effects.csv", effect_rows(data))
    conditions = []
    for campaign in ("a", "c", "d"):
        for key, result in data[campaign]["reports"].items():
            arm, model = key.split("/")
            for policy, r in result["conditions"].items():
                conditions.append(dict(campaign=campaign, arm=arm, model=model, policy=policy,
                    clean_n=r["clean_success"]["n"], clean_correct=r["clean_success"]["count"],
                    fault_n=r["fault_success"]["n"], fault_correct=r["fault_success"]["count"],
                    unsafe_action_n=r["action_required_unsafe_state"]["n"],
                    unsafe_action_count=r["action_required_unsafe_state"]["count"],
                    mean_fault_tokens=r["mean_fault_tokens"], mean_fault_raw_calls=r["mean_fault_raw_calls"],
                    pooled_view=arm == "sampling_pooled"))
    csv_file(out / "supplement_raw_counts.csv", conditions)
    b_rows = [dict(model=m, policy=c, **{k: r[k] for k in (
        "correct_clean", "observed_clean", "correct_fault", "observed_fault", "mean_fault_tokens", "mean_fault_raw_calls")})
        for m, v in data["b"].items() for c, r in v["conditions"].items()]
    csv_file(out / "external_raw_counts.csv", b_rows)
    primary_rows = [dict(model=m, policy=c, clean_n=r["clean_success"]["n"],
        clean_correct=r["clean_success"]["count"], fault_n=r["fault_success"]["n"],
        fault_correct=r["fault_success"]["count"], unsafe_action_n=r["persistent_unsafe_action_required"]["n"],
        unsafe_action_count=r["persistent_unsafe_action_required"]["count"])
        for m, v in data["primary"]["models"].items() for c, r in v["conditions"].items()]
    csv_file(out / "primary_raw_counts.csv", primary_rows)
    captions = '''# English figures and suggested captions

All figures use final audited results. PNG is suitable for slides; PDF and SVG
are vector formats. ALL_FIGURES_EN.pdf contains the five figures in order.
CSV files retain raw denominators and paired-effect eligibility/weighting.
No model was rerun and no frozen experimental output was modified.

## 1. Six tool-failure classes and MedGuard recovery

MedGuard verifies clinical-tool evidence and ambiguous writes through public
interfaces. The table pairs each injected failure class with its recovery
mechanism. Write attribution assumes a single writer and immediately visible
commits. This taxonomy is a description of the evaluated implementation, not a
claim that every failure can be recovered. B-lite tests only read-side failures.

## 2. Primary study: paired recovery effects

Points show MedGuard-minus-vanilla safe-success differences, with 95% percentile
intervals from 10,000 patient-cluster bootstrap draws. Grey shows all 255 test
tasks; teal shows each system's vanilla-clean-capable subset, the primary
endpoint. The n labels describe the latter subset. Faults are averaged within
tasks. Nine configurations represent eight checkpoints, not nine independent
model families. Historical Gemma results remain; GPT-OSS was added in follow-ups.

## 3. Controls, equal budgets and safety trade-offs

Raw fault safe-success rates are shown for matched controls (806 assignments
per policy/model) and the strict 12-raw-call condition (160 assignments). Informed
agents receive explicit verification and recovery guidance. The third panel
separately shows unsafe final states among action-required assignments (202 per
policy/model). Higher task success does not guarantee fewer unsafe states, as
the Llama counterexample illustrates. A shared budget ceiling does not imply
equal realized computation. These descriptive bars do not have confidence
intervals and must not be treated as patient-harm estimates.

## 4. External B-lite validation

The left panel shows raw answer correctness across 500 fault assignments per
policy/model. The right panel shows paired MedGuard-minus-vanilla and
MedGuard-minus-informed effects, weighting patients equally and averaging
questions and assigned faults within patients. Intervals use 10,000 patient
bootstrap draws. All four MedGuard-minus-informed intervals cross zero.
The 100-question sample covers 45 previously unattempted B patients and 25
templates; it is not a newly sampled independent hospital cohort. All four
models completed 1,800 episodes each. This is read-only QA, not write safety.

## 5. Task expansion and boundaries

C evaluates semi-synthetic decision branches and external-source operational
rules. Raw safe-success rates and parent-task-weighted paired effects are
reported separately. Qwen's external-action regression is retained, as are
weak-model and unsafe-state counterexamples. Delayed write visibility remains
a complete failure in the tested MedGuard boundary arm. These are not authentic
treatment-policy or clinical-outcome evaluations.

## Statistical and presentation notes

Safe success requires correct answers, required state changes and no forbidden
collateral mutations. Unsafe states and answer correctness are distinct
endpoints. Episode counts are not independent patient counts. No confirmatory
p-values or multiplicity-adjusted significance claims are provided. Do not
infer model rankings under identical protocols: model-specific adapters and
historical execution strata are documented in the full result report.
'''
    (out / "CAPTIONS_EN.md").write_text(captions)
    gallery = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>MedGuard: final figures</title>',
               '<style>body{max-width:1500px;margin:32px auto;padding:0 24px;font-family:system-ui;color:#153344}img{width:100%;height:auto}section{margin:42px 0;border-top:1px solid #ddd}a{color:#007d88}</style>',
               '<h1>MedToolBench / MedGuard — final experimental figures</h1>',
               '<p><a href="ALL_FIGURES_EN.pdf">All figures (PDF)</a> · <a href="CAPTIONS_EN.md">Captions and interpretation notes</a></p>']
    for image in sorted(out.glob("0*.png")):
        gallery.append(f'<section><h2>{image.stem.replace("_", " ")}</h2><p><a href="{image.stem}.pdf">PDF</a> · <a href="{image.stem}.svg">Editable SVG</a> · <a href="{image.name}">PNG</a></p><img src="{image.name}" alt="{image.stem.replace("_", " ")}"></section>')
    (out / "index.html").write_text("\n".join(gallery) + "</html>\n")
    evidence.unchanged()
    script = Path(__file__).resolve()
    manifest = dict(formal_episodes=data["counts"], total_formal=127861,
        audited_run_directories=len(evidence.runs), inputs=evidence.inputs,
        script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
        evidence_module_sha256=hashlib.sha256(Path(__import__("medtoolbench.publication.evidence", fromlist=["__file__"]).__file__).read_bytes()).hexdigest(),
        matplotlib_version=matplotlib.__version__, source_data_unchanged=True,
        outputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()},
        interpretation="Descriptive effects; no confirmatory p-values, independent hospital cohort or clinical harm claims")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(out), "figures": 5, "formats": ["png", "svg", "pdf"], "source_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()
