"""Candidate model-output envelope extraction; not general JSON-file repair.

Locate the first top-level object, respecting strings, escapes and nesting,
then strictly decode the exact substring. Never repair JSON, skip an invalid
first object, unwrap arrays, scan reasoning, or choose an action by its score.
Only an explicit revision may install this adapter; frozen runners stay intact.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import json

from medtoolbench.run_local import parse_action as _legacy_parse
from medtoolbench.research.input_validation import validate_body
from .input_guard import validate_meta

REVISION = 'first-object-envelope-v2'


@dataclass(frozen=True)
class Envelope:
    text: str
    value: dict
    start: int
    end: int
    reasoning_prefix_removed: bool


def _first_start(raw):
    """Read only the envelope before the first structured value.

Do not scan an already selected action's strings or hallucinated trailing
conversation for a later </think>. Only a leading reasoning block is stripped.
Unmarked reasoning whose opening tag was in a prompt requires an explicit
channel decoder upstream; arbitrary earlier JSON is never skipped to find a
more convenient action after a trailing marker.
"""
    quoted, escaped, i, reasoning_removed = False, False, 0, False
    while i < len(raw):
        char = raw[i]
        if quoted:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                quoted = False
            i += 1
            continue
        if char == '"':
            quoted = True
        elif char == '[':
            raise ValueError('First structured value is an array, not an action object')
        elif char == '{':
            return i, reasoning_removed
        elif raw.startswith('<|', i) and raw.find('|>', i + 2) != -1:
            raise ValueError('Native message framing requires its own channel decoder; no JSON scanning')
        elif raw.startswith('<think>', i):
            if reasoning_removed:
                raise ValueError('Ambiguous repeated reasoning prefix; no action dispatched')
            end = raw.find('</think>', i + len('<think>'))
            nested = raw.find('<think>', i + len('<think>'))
            if end == -1 or nested != -1 and nested < end:
                raise ValueError('Incomplete or ambiguous reasoning block; no action dispatched')
            reasoning_removed = True
            i = end + len('</think>')
            continue
        elif raw.startswith('</think>', i):
            if reasoning_removed:
                raise ValueError('Ambiguous repeated reasoning prefix; no action dispatched')
            reasoning_removed = True
            i += len('</think>')
            continue
        i += 1
    raise ValueError('No top-level JSON object in designated output')


def extract_object(raw):
    if not isinstance(raw, str):
        raise ValueError('Model output must be text')
    start, reasoning_removed = _first_start(raw)
    stack, quoted, escaped = [], False, False
    for i in range(start, len(raw)):
        char = raw[i]
        if quoted:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char in '{[':
            stack.append(char)
        elif char in '}]':
            expected = '{' if char == '}' else '['
            if not stack or stack.pop() != expected:
                raise ValueError('Unbalanced first JSON object; no fallback to a later object')
            if not stack:
                text = raw[start:i + 1]
                # Same strict string syntax as the original decoder. In
                # particular, literal newlines inside quoted code still fail.
                value = json.loads(text)
                if not isinstance(value, dict):
                    raise ValueError('Expected one action object')
                return Envelope(text, value, start, i + 1, reasoning_removed)
    raise ValueError('Incomplete first JSON object; no fallback to a nested/later object')


def parse_legacy(raw):
    return _legacy_parse(extract_object(raw).text)


def parse_fhir_body(raw):
    action = parse_legacy(raw)
    if 'final' not in action:
        validate_body(action.get('body'))
    return action


def parse_fhir(raw):
    action = parse_fhir_body(raw)
    validate_meta(action)
    return action


def parse_external(raw):
    action = extract_object(raw).value
    # Preserve the frozen external runner's action validation, without scanning
    # the extracted body's strings again for reasoning markers.
    if 'final' in action:
        if set(action) != {'final'} or not isinstance(action['final'], dict):
            raise ValueError('Final cannot also contain a tool action')
    elif action.get('tool') not in {'metadata', 'get_resources', 'execute_python_code'}:
        raise ValueError('Unsupported read-only tool')
    return action


@contextmanager
def parser_scope():
    """Explicit development scope; restores every original parser on exit.

Production launchers do NOT enter this scope. Adoption requires a separately
recorded experiment revision, never a hot patch to existing evidence.
"""
    from unittest.mock import patch
    from medtoolbench import run_local
    from medtoolbench.research import protocol, run
    from . import input_guard, external_runner
    record = {'revision': REVISION, 'first_object_only': True,
        'json_payload_repaired': False, 'reasoning_executable': False,
        'file_manifest_json_unchanged': True, 'formal_dispatch_enabled': False}
    with patch.object(run_local, 'parse_action', parse_legacy), \
         patch.object(protocol, 'parse_action', parse_fhir_body), \
         patch.object(run, 'parse_action', parse_fhir), \
         patch.object(input_guard, 'original_parse_action', parse_fhir_body), \
         patch.object(input_guard, 'parse_action', parse_fhir), \
         patch.object(external_runner, 'parse', parse_external):
        yield record
