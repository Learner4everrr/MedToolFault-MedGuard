"""Declared, native Harmony transport for A/C/D only; no clinical repair.

The original public messages remain in the episode. A versioned transport
instruction and tool schema are added only when serializing model input.
Valid calls retain reasoning, recipient, arguments, handoff and tool feedback.
Malformed frames are rejected, never searched for a convenient JSON object.
"""
import copy
import json
import re

from .gptoss_model import MESSAGE
from .gptoss_protocol import _object

VERSION = 'harmony-declared-fhir-v3'
TOOL = 'fhir_request'
RECIPIENT = 'functions.' + TOOL
INSTRUCTION = '''\nTRANSPORT ADAPTER (supersedes only the JSON delivery/feedback convention above):
Send each HTTP action object as arguments to functions.fhir_request, using the
native commentary-channel tool handoff. Exactly one call per turn. Its response
is returned as a tool message with status and body, not a user message.
Do not emit HTTP actions in the final channel. To finish, emit the original
{"final":{...}} answer object in the final channel. All other instructions,
clinical fields, public information and budgets above remain unchanged.
'''
TOOLS = [{'type': 'function', 'function': {
    'name': TOOL, 'description': 'Execute one HTTP action in the owned research FHIR sandbox. Returns status and body.',
    'parameters': {'type': 'object', 'properties': {
        'method': {'type': 'string', 'enum': ['GET', 'POST', 'PUT', 'DELETE']},
        'path': {'type': 'string', 'description': 'Relative FHIR path, including any query parameters.'},
        'body': {'type': 'object', 'description': 'The unmodified JSON request body, when required.'}},
        'required': ['method', 'path'], 'additionalProperties': False}}}]


def header(raw):
    pieces = raw.split('<|constrain|>')
    if len(pieces) > 2 or len(pieces) == 2 and pieces[1].strip() != 'json':
        raise ValueError('unsupported constrained format')
    parts = pieces[0].split('<|channel|>')
    if len(parts) != 2:
        raise ValueError('explicit channel required')
    role, channel = parts[0].split(), parts[1].split()
    if not role or role[0] != 'assistant' or not channel or channel[0] not in ('analysis', 'commentary', 'final'):
        raise ValueError('unsupported role/channel')
    extras = role[1:] + channel[1:]
    recipients = [w[3:] for w in extras if w.startswith('to=')]
    formats = [w for w in extras if w in ('json', 'code')]
    if len(recipients) > 1 or len(formats) > 1 or (len(pieces) == 2 and formats):
        raise ValueError('duplicate recipient/format')
    if any(w not in ('json', 'code') and not w.startswith('to=') for w in extras):
        raise ValueError('unsupported header field')
    if recipients and not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.]*', recipients[0]):
        raise ValueError('malformed recipient')
    return channel[0], recipients[0] if recipients else None


def decode(raw):
    metadata = {'version': VERSION, 'kind': 'rejected', 'frames': [], 'arguments_modified': False}
    stream, position, chosen = '<|start|>assistant' + raw, 0, None
    try:
        for match in MESSAGE.finditer(stream):
            if match.start() != position:
                raise ValueError('malformed frame boundary')
            position = match.end()
            h, body, ending = match.groups()
            channel, recipient = header(h)
            if re.search(r'<\|(?:start|channel|message|constrain)\|>', body):
                raise ValueError('unexpected control token in body')
            metadata['frames'].append({'channel': channel, 'recipient': recipient, 'body': body, 'ending': ending})
            if chosen is not None:
                raise ValueError('data after terminal action')
            if recipient is not None:
                if recipient != RECIPIENT:
                    raise ValueError('undeclared recipient; use ' + RECIPIENT)
                if channel != 'commentary' or ending != '<|call|>':
                    raise ValueError('FHIR call requires commentary and explicit handoff')
                arguments = _object(body)
                metadata['arguments'] = arguments
                if 'final' in arguments or set(arguments) - {'method', 'path', 'body'}:
                    raise ValueError('Tool arguments must contain only method, path and optional body')
                if not {'method', 'path'} <= set(arguments):
                    raise ValueError('FHIR tool requires both method and path')
                # Structural parsing only. Original action/body validators decide
                # malformed fields; there is no inference of missing arguments.
                chosen = body
                metadata.update(kind='tool', recipient=recipient, arguments=arguments)
            elif channel == 'final':
                if ending != '<|return|>':
                    raise ValueError('final answer requires end-of-turn token')
                _object(body)
                from medtoolbench.research.protocol import parse_action
                action = parse_action(body)
                if set(action) != {'final'}:
                    raise ValueError('HTTP actions require a native FHIR tool handoff')
                chosen = body
                metadata.update(kind='final')
            elif ending != '<|end|>':
                raise ValueError('nonterminal content cannot execute')
            if ending in ('<|return|>', '<|call|>', '<|endoftext|>') and position != len(stream):
                raise ValueError('trailing data after terminal')
        if position != len(stream) or chosen is None:
            raise ValueError('no complete action')
        return chosen, 'native_' + metadata['kind'], metadata
    except (ValueError, TypeError, KeyError, IndexError, RecursionError) as error:
        metadata.update(kind='rejected', error=str(error))
        return '', 'protocol_rejected', metadata


def native_messages(messages, schemas=True, history=True):
    """Pure transformation: no mutation of clinical prompts or saved evidence."""
    if len(messages) < 2 or messages[0]['role'] != 'system' or messages[1]['role'] != 'user':
        raise ValueError('Expected the original public prompt prefix')
    result = copy.deepcopy(messages[:2])
    if schemas:
        result[0]['content'] += INSTRUCTION
    waiting = False
    for message in messages[2:]:
        native = message.get('native_transport')
        if message['role'] == 'assistant':
            if waiting:
                raise ValueError('Missing tool feedback')
            if native is None or native.get('version') != VERSION:
                raise ValueError('Unversioned assistant history')
            if history and native['kind'] == 'tool':
                reasoning = '\n'.join(f['body'] for f in native['frames'] if f['recipient'] is None)
                result.append({'role': 'assistant', 'thinking': reasoning, 'tool_calls': [
                    {'type': 'function', 'function': {'name': TOOL, 'arguments': copy.deepcopy(native['arguments'])}}]})
                waiting = True
            else:
                # Rejected output is retained verbatim in generation evidence,
                # not inserted as potentially unterminated Harmony control text.
                result.append({'role': 'assistant', 'content': message['content']})
        elif message['role'] == 'user':
            if waiting:
                result.append({'role': 'tool', 'name': TOOL, 'content': json.loads(message['content'])})
                waiting = False
            else:
                result.append({'role': 'user', 'content': message['content']})
        else:
            raise ValueError('Unexpected canonical history role')
    if waiting:
        raise ValueError('Tool feedback not yet available')
    return result


def advance(episode, generation):
    """Run the original clinical executor; annotate history after it returns."""
    from medtoolbench.research.run import Episode
    text, status, metadata = decode(generation['raw_harmony'])
    if (text, status, metadata) != (generation['text'], generation['harmony_status'], generation['native_transport']):
        raise RuntimeError('Native generation evidence mismatch')
    index = len(episode.messages)
    Episode.advance(episode, generation)
    episode.messages[index]['native_transport'] = copy.deepcopy(metadata)
    if metadata['kind'] == 'rejected':
        # Accurate transport feedback, not a clinical hint or argument repair.
        episode.messages[-1]['content'] = json.dumps({'status': 400, 'body': {
            'error': metadata['error'], 'instruction': 'Use the declared native FHIR tool or the final answer protocol.'}}, separators=(',', ':'))


def render(tokenizer, template, messages, schemas=True, history=True):
    from medtoolbench.research.models import input_ids
    from .gptoss_model import REASONING
    return input_ids(tokenizer.apply_chat_template(native_messages(messages, schemas, history),
        chat_template=template, tools=TOOLS if schemas else None, reasoning_effort=REASONING,
        tokenize=True, return_dict=False, add_generation_prompt=True))
