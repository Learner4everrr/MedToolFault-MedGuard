"""Resolve final evidence by its audited lineage, not directory timestamps."""
import hashlib
import json
from pathlib import Path


MODELS = ("qwen35_9b", "ministral3_8b", "llama31_8b", "gptoss20b")
LABELS = {
    "qwen35_9b": "Qwen3.5-9B", "ministral3_8b": "Ministral-3-8B",
    "llama31_8b": "Llama-3.1-8B", "gptoss20b": "GPT-OSS-20B",
    "qwen25_7b": "Qwen2.5-7B", "qwen25_14b": "Qwen2.5-14B",
    "qwen3_8b": "Qwen3-8B (non-thinking)",
    "qwen3_8b_thinking": "Qwen3-8B (thinking)",
    "gemma3_12b": "Gemma-3-12B", "phi4_mini": "Phi-4-mini",
}
ACD_ROOT = "data/gptoss_compile_recovery_v1"
B_ROOTS = {
    "qwen35_9b": "data/b_lite_v1",
    "ministral3_8b": "data/b_lite_remaining_models_v1",
    "llama31_8b": "data/b_lite_llama_eight_gpu_v1",
    "gptoss20b": "data/b_lite_gptoss_env_v1",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False).encode())


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Evidence:
    def __init__(self, project):
        self.project = Path(project).resolve()
        self.inputs = {}
        self.runs = set()

    def path(self, path):
        path = Path(path)
        path = (path if path.is_absolute() else self.project / path).resolve()
        require(path.is_relative_to(self.project), "Evidence outside project: " + str(path))
        return path

    def read(self, path):
        path = self.path(path)
        raw = path.read_bytes()
        name = str(path.relative_to(self.project))
        checksum = sha(raw)
        require(name not in self.inputs or self.inputs[name] == checksum,
                "Input changed while reading: " + name)
        self.inputs[name] = checksum
        return json.loads(raw)

    def audit(self, run, checksum=None, encoding="json"):
        run = self.path(run)
        require(run not in self.runs, "Repeated evidence directory: " + str(run))
        self.runs.add(run)
        audit = self.read(run / "audit.json")
        require(audit.get("all_pass") is True, "Failed shard audit: " + str(run))
        if checksum:
            got = digest(audit) if encoding == "json" else self.inputs[
                str((run / "audit.json").relative_to(self.project))]
            require(got == checksum, "Audit hash mismatch: " + str(run))
        return audit

    def unchanged(self):
        for name, expected in self.inputs.items():
            require(sha((self.project / name).read_bytes()) == expected,
                    "Input changed during analysis: " + name)

    def load(self):
        root = Path("data/primary_study_v1/report")
        primary = self.read(root / "metrics.json")
        receipt = self.read(root / "complete.json")
        require(self.inputs[str(root / "metrics.json")] == receipt["metrics_sha256"],
                "Primary report hash mismatch")
        require(primary["episodes"] == receipt["episodes"] == 47745, "Primary count")
        for ref in primary["provenance"].values():
            proof = self.audit(ref["run"])
            require(proof["episodes"] == ref["episodes"], "Primary shard count")
        data = {"primary": primary}
        counts = {"primary": 47745}
        for arm, expected in (("a", 50212), ("c", 19344), ("d", 3360)):
            root = Path(ACD_ROOT) / ("combined_" + arm) / "report"
            metrics = self.read(root / "metrics.json")
            receipt = self.read(root / "complete.json")
            require(receipt["all_pass"] and receipt["plan_hash"] == metrics["plan_hash"],
                    "Supplement completion mismatch")
            require(digest(self.path(root / "metrics.json").read_text()) == receipt["metrics_hash"],
                    "Supplement report hash mismatch")
            # sampling_pooled is a view of three seeds, not new observations.
            count = sum(v["episodes"] for k, v in metrics["reports"].items()
                        if not k.startswith("sampling_pooled/"))
            require(count == receipt["episodes_test"] == expected, "Supplement count")
            for ref in metrics["runs"]:
                self.audit(ref["run"], ref["audit_hash"])
            data[arm], counts[arm] = metrics, count
        data["b"] = {}
        for model, name in B_ROOTS.items():
            root = Path(name)
            metrics = self.read(root / "report/metrics.json")
            result = metrics["models"][model]
            require(result["complete"] and result["missing"] == 0 and
                    result["audited"] == result["assigned"] == 1800, "B-lite incomplete")
            count = 0
            for ref in metrics["audit_references"]:
                if model not in ref["job"]:
                    continue
                job = self.read(root / "jobs" / ref["job"] / "result.json")
                require(job["state"] == "complete", "B-lite unresolved shard")
                proof = self.audit(job["run"], ref["audit_sha256"], encoding="bytes")
                require(proof["episodes"] == job["episodes"], "B-lite shard count")
                count += job["episodes"]
            require(count == 1800, "B-lite reference count")
            data["b"][model] = result
        counts["b_lite"] = 7200
        require(sum(counts.values()) == 127861 and len(self.runs) == 826,
                "Final evidence inventory mismatch")
        data["counts"] = counts
        return data


def effect_rows(data):
    """All effects retain their endpoint, weighting, stratum and eligibility."""
    rows = []
    for model, result in data["primary"]["models"].items():
        for field, eligibility in (("paired_guard_gain", "all_tasks"),
                                   ("paired_guard_gain_clean_capable", "vanilla_clean_capable")):
            effect = result[field]
            rows.append(dict(campaign="primary", arm="primary", model=model,
                endpoint="safe_success", eligibility=eligibility,
                comparison="medguard-vanilla", difference=effect["task_weighted_gain"],
                ci_lower=effect["patient_cluster_ci95"][0], ci_upper=effect["patient_cluster_ci95"][1],
                parents=effect["tasks"], patients=effect["groups"], weighting="task_equal"))
    for campaign in ("a", "c", "d"):
        for key, result in data[campaign]["reports"].items():
            arm, model = key.split("/")
            for field, endpoint, eligibility in (
                    ("comparisons", "safe_success", "all_tasks"),
                    ("conditional_comparisons", "safe_success", "vanilla_clean_capable"),
                    ("unsafe_comparisons", "unsafe_state", "action_required")):
                for comparison, effect in result[field].items():
                    if effect["difference"] is None:
                        continue
                    rows.append(dict(campaign=campaign, arm=arm, model=model,
                        endpoint=endpoint, eligibility=eligibility, comparison=comparison,
                        difference=effect["difference"], ci_lower=effect["ci95"][0],
                        ci_upper=effect["ci95"][1], parents=effect["parents"], patients=effect["groups"],
                        weighting="parent_equal", pooled_view=arm == "sampling_pooled"))
    for model, result in data["b"].items():
        for key, effect in result["comparisons"].items():
            comparison, _, subset = key.partition("/")
            rows.append(dict(campaign="b_lite", arm="external_read_only", model=model,
                endpoint="answer_correctness", eligibility=subset or "all_questions",
                comparison=comparison, difference=effect["difference"],
                ci_lower=effect["ci95"][0], ci_upper=effect["ci95"][1],
                patients=effect["patients"], weighting="patient_equal"))
    return rows
