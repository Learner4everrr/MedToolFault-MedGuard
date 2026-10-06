"""Versioned, policy-independent syntax guard; never repairs a model's payload.

The frozen protocol already validates JSON bodies. This adds only the missing
top-level FHIR meta type check, before any policy-specific pre-write reads.
The original Episode.advance records the generated text and a 400 response.
No raw request is issued, so no raw-call budget is charged for parser rejection.
"""
from medtoolbench.research.protocol import parse_action as original_parse_action


def validate_meta(action):
    if "final" in action:
        return
    body = action.get("body")
    if isinstance(body, dict) and "meta" in body and not isinstance(body["meta"], dict):
        raise ValueError("FHIR body.meta must be a JSON object when present; omit it when unspecified. The submitted body was not modified or dispatched.")


def parse_action(raw):
    action = original_parse_action(raw)
    validate_meta(action)
    return action


def install():
    from medtoolbench.research import run
    run.parse_action = parse_action
    return {"revision": "meta-object-v1", "policy_independent": True,
            "invalid_body_corrected": False, "rejection_stage": "action_parser",
            "rejected_actions_consume_turn_but_no_raw_call": True}
