"""Versioned Harmony-to-JSON transport; never an alternative tool executor.

Only explicit assistant tool handoffs or final-channel messages can become
actions. Ordinary reasoning is never searched for JSON. The clinical parser,
tool feedback roles, prompts, budgets, sandbox and grading remain unchanged.
Native FHIR recipients are a fixed transport alias allowlist, not Python
attributes to resolve. External calls can only name the three public tools.
"""
import json
import re

from .gptoss_model import MESSAGE, extract_final

VERSION = 'harmony-json-bridge-v2'
EXTERNAL = frozenset(('metadata', 'get_resources', 'execute_python_code'))
FHIR = {'repo_browser.' + name: verb for name, verb in (
    ('send_request', None), ('request', None), ('get', 'GET'), ('post', 'POST'),
    ('put', 'PUT'), ('delete', 'DELETE'))}


def _object(body):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def constant(value):
        raise ValueError('non-finite JSON constant')
    value = json.loads(body, object_pairs_hook=pairs, parse_constant=constant)
    if not isinstance(value, dict):
        raise ValueError('tool arguments must be exactly one object')
    return value


def _header(raw):
    pieces = raw.split('<|constrain|>')
    if len(pieces) > 2 or len(pieces) == 2 and pieces[1].strip() != 'json':
        raise ValueError('unsupported constrained format')
    pieces = pieces[0].split('<|channel|>')
    if len(pieces) != 2:
        raise ValueError('channel required')
    role, channel = pieces[0].split(), pieces[1].split()
    if not role or role[0] != 'assistant' or not channel:
        raise ValueError('assistant role required')
    if channel[0] not in ('analysis', 'commentary', 'final'):
        raise ValueError('unsupported channel')
    extras = role[1:] + channel[1:]
    recipients = [word[3:] for word in extras if word.startswith('to=')]
    if len(recipients) > 1 or any(not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.]*', v) for v in recipients):
        raise ValueError('ambiguous recipient')
    if any(word != 'code' and not word.startswith('to=') for word in extras):
        raise ValueError('unsupported header field')
    if extras.count('code') > 1:
        raise ValueError('duplicate format marker')
    return channel[0], recipients[0] if recipients else None


def _tool(body, recipient, mode):
    action = _object(body)
    name = recipient.removeprefix('functions.')
    if mode == 'fhir':
        if name not in FHIR:
            raise ValueError('undeclared FHIR transport alias')
        if not {'method', 'path'} <= action.keys() or set(action) - {'method', 'path', 'body'}:
            raise ValueError('native FHIR body is not a canonical action')
        if not isinstance(action['method'], str) or not isinstance(action['path'], str):
            raise ValueError('method/path must remain strings')
        if FHIR[name] is not None and action['method'] != FHIR[name]:
            raise ValueError('recipient and action method disagree')
        # The original action parser and client still validate every argument.
        return body, 'native_fhir', []
    if mode != 'external' or name not in EXTERNAL:
        raise ValueError('undeclared external tool')
    if 'final' in action or 'method' in action or 'path' in action:
        raise ValueError('mixed action protocols')
    if 'tool' in action:
        if action['tool'] != name:
            raise ValueError('recipient and tool selector disagree')
        return body, 'native_external', []
    # Only inject the missing selector. All argument values, including code
    # escapes/imports/errors, are preserved; no argument or answer repair.
    return json.dumps({'tool': name, **action}, ensure_ascii=False), 'native_external', ['tool']


def decode(raw, mode):
    if mode not in ('fhir', 'external'):
        raise ValueError('explicit protocol mode required')
    legacy, status = extract_final(raw)
    if status == 'final':
        return legacy, status, {'version': VERSION, 'mode': mode, 'added_fields': []}
    stream = '<|start|>assistant' + raw
    position, selected, metadata = 0, None, {'version': VERSION, 'mode': mode}
    try:
        for match in MESSAGE.finditer(stream):
            if match.start() != position:
                raise ValueError('malformed frame boundary')
            header, body, ending = match.groups()
            position = match.end()
            channel, recipient = _header(header)
            if re.search(r'<\|(?:start|channel|message|constrain)\|>', body):
                raise ValueError('unexpected control token in body')
            if ending in ('<|return|>', '<|call|>', '<|endoftext|>') and position != len(stream):
                raise ValueError('trailing data after terminal')
            if selected is not None:
                raise ValueError('multiple terminal actions')
            if recipient is not None:
                if channel == 'final' or ending != '<|call|>':
                    raise ValueError('recipient requires an explicit non-final handoff')
                text, status, added = _tool(body, recipient, mode)
                selected = (text, status)
                metadata.update(recipient=recipient, channel=channel, added_fields=added)
            elif ending == '<|call|>':
                raise ValueError('handoff requires recipient')
            elif channel == 'final':
                selected = (body, 'final')
                metadata.update(channel=channel, added_fields=[])
            elif ending != '<|end|>':
                raise ValueError('reasoning/commentary without a terminal action')
        if position != len(stream) or selected is None:
            raise ValueError('no unique action')
        return *selected, metadata
    except (ValueError, TypeError, RecursionError) as error:
        metadata['error'] = str(error)
        return '', 'protocol_rejected', metadata
