"""Descriptive paired analysis with task-cluster bootstrap, not clinical inference."""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def mean(values):
    return sum(values) / len(values) if values else None


def bootstrap(values, seed=20260911, samples=4000):
    if not values:
        return [None, None]
    rng = random.Random(seed)
    estimates = sorted(mean(rng.choices(values, k=len(values))) for _ in range(samples))
    return [estimates[int(samples * .025)], estimates[int(samples * .975)]]


def summarize(rows):
    result = {}
    for condition in sorted({r["condition"] for r in rows}):
        subset = [r for r in rows if r["condition"] == condition]
        clean = {r["task_id"]: r for r in subset if r["hazard"] == "clean"}
        hazards = [r for r in subset if r["hazard"] != "clean"]
        groups = defaultdict(list)
        for r in subset:
            groups[r["hazard"]].append(r)
        by_hazard = {}
        for hazard, group in sorted(groups.items()):
            paired = [r for r in group if r["task_id"] in clean]
            capable = [r for r in paired if clean[r["task_id"]]["grade"]["success"]]
            triggered = [r for r in group if r["triggered"]]
            actions = [r for r in group if r["action_task"]]
            by_hazard[hazard] = {"n": len(group), "successes": sum(r["grade"]["success"] for r in group),
                "success_rate": mean([r["grade"]["success"] for r in group]),
                "paired_clean_success_rate": mean([clean[r["task_id"]]["grade"]["success"] for r in paired]),
                "conditional_robustness_n": len(capable),
                "conditional_robustness": mean([r["grade"]["success"] for r in capable]),
                "hazard_triggered_n": len(triggered),
                "recovery_success_rate": mean([r["grade"]["success"] for r in triggered]),
                "unsafe_n": sum(bool(r["grade"]["unsafe"]) for r in group),
                "action_episode_n": len(actions),
                "unsafe_mutation_rate": mean([bool(r["grade"]["unsafe"]) for r in actions]),
                "duplicate_n": sum("duplicate" in r["grade"]["unsafe"] for r in group),
                "duplicate_rate_among_triggered": mean(["duplicate" in r["grade"]["unsafe"] for r in triggered]),
                "tool_calls_mean": mean([r["tool_calls"] for r in group]),
                "tokens_mean": mean([r["total_tokens"] for r in group]),
                "paired_tokens_delta_mean": mean([r["total_tokens"] - clean[r["task_id"]]["total_tokens"] for r in paired])}
        per_task_gap = defaultdict(list)
        for r in hazards:
            if r["task_id"] in clean:
                per_task_gap[r["task_id"]].append(int(clean[r["task_id"]]["grade"]["success"]) - int(r["grade"]["success"]))
        gaps = [mean(v) for v in per_task_gap.values()]
        result[condition] = {"by_hazard": by_hazard,
            "task_weighted_paired_gap": mean(gaps), "task_cluster_bootstrap_95ci": bootstrap(gaps),
            "hazard_episode_success_rate": mean([r["grade"]["success"] for r in hazards]),
            "total_episodes": len(subset), "incomplete_pair_count": sum(r["task_id"] not in clean for r in hazards)}
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    all_results = {}
    lines = ["# Local synthetic mechanism pilot", "",
        "Development results, not MAB-v3 reproduction or clinical deployment evidence.", "",
        "Rates below require correct final answer AND correct persistent state AND no unsafe mutation.", "",
        "| Model | Condition | Hazard | Safe success | Triggered | Unsafe / action episodes | Duplicate | Mean calls | Mean tokens |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for run in args.runs:
        if not (run / "complete.json").exists():
            raise SystemExit(f"Incomplete run: {run}")
        rows = [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines()]
        if len({r["key"] for r in rows}) != len(rows):
            raise ValueError("Duplicate episode keys")
        config = json.loads((run / "config.json").read_text())
        complete = json.loads((run / "complete.json").read_text())
        assert len(rows) == complete["completed"]
        expected_tasks = json.loads(Path(config["manifest"]).read_text())
        if config.get("limit_tasks"):
            expected_tasks = expected_tasks[:config["limit_tasks"]]
        expected = {f"{t['id']}--{h}--{c}" for t in expected_tasks for h in t["hazards"] for c in config["conditions"]}
        assert {r["key"] for r in rows} == expected, "Run does not cover the declared matrix"
        summary = summarize(rows)
        all_results[config["name"]] = {"run": str(run), "summary": summary, "config": config}
        for condition, results in summary.items():
            for h, s in results["by_hazard"].items():
                lines.append(f"| {config['name']} | {condition} | {h} | {s['successes']}/{s['n']} | "
                    f"{s['hazard_triggered_n']} | {s['unsafe_n']}/{s['action_episode_n']} | {s['duplicate_n']} | "
                    f"{s['tool_calls_mean']:.2f} | {s['tokens_mean']:.0f} |")
    lines += ["", "## Paired uncertainty", "",
        "Average the clean-minus-hazard success difference within each task, then average across tasks. "
        "95% percentile bootstrap resamples underlying tasks (4,000 draws), not fault variants. "
        "This remains only four synthetic task templates; intervals do not establish clinical generalization.", ""]
    for model, value in all_results.items():
        for c, s in value["summary"].items():
            lo, hi = s["task_cluster_bootstrap_95ci"]
            lines.append(f"- {model}, {c}: paired gap {100*s['task_weighted_paired_gap']:.1f} pp "
                         f"(95% bootstrap CI {100*lo:.1f}–{100*hi:.1f}).")
    lines += ["", "## Important limits", "",
        "- Synthetic fixtures and explicit payload templates simplify task/format competence; no MAB-v3 patient claims.",
        "- Two model sizes from one older model family are not broad/frontier-model evidence.",
        "- Four single-shot hazard classes; two timeout outcomes are variants of one class.",
        "- HAPI smoke validates backend mechanisms separately; these LLM rollouts use SQLite replay.",
        "- MedGuard always verifies matching observation searches, increasing tool cost even when clean.",
        "- Exact-payload write equivalence does not yet cover all FHIR semantic equivalences or concurrent writers.",
        "- Silent corruption acceptance is not equated with every failed episode; inspect saved traces.",
        "- No patient-level human labels, no model APIs, no clinical effectiveness or incident-frequency estimates.",
        "- Smoke runs affected by harness debugging are excluded. Full pilot is development, not a held-out test."]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(all_results, indent=2) + "\n")
    (args.out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
