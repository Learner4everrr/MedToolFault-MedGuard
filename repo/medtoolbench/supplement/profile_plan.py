"""Validate that a C-only implementation repair leaves the experiment intact."""
from medtoolbench.storage import digest

IMPLEMENTATION_FIELDS = {"plan_hash", "version", "created_at", "session", "resource_handoff",
    "runner_module", "source_hashes", "profile_correction"}


def experiment_identity(plan):
    return digest({k: v for k, v in plan.items() if k not in IMPLEMENTATION_FIELDS})


def validate_repair(parent, revised):
    if experiment_identity(parent) != experiment_identity(revised):
        raise RuntimeError("Profile repair changed tasks, labels, models, budgets or experiment design")
    if revised.get("runner_module") != "medtoolbench.supplement.profile_runner":
        raise RuntimeError("Unexpected corrected runner")
    if any(revised["source_hashes"].get(name) != value for name, value in parent["source_hashes"].items()):
        raise RuntimeError("An existing frozen implementation fingerprint was changed")
    correction = revised["profile_correction"]
    if correction["parent_plan_hash"] != parent["plan_hash"] or correction["experiment_identity"] != experiment_identity(parent):
        raise RuntimeError("Profile-repair provenance does not match parent")
    return True
