"""Versioned rejection handling, not correction of an agent's clinical payload.

The frozen concurrent writer copies the requested body. A 4xx rejection of that
copy must remain a recorded tool rejection, not crash the whole experiment.
Do not commit, mark the fault consumed, retry, or repair the submitted body.
Server errors and unexpected exceptions still fail closed as infrastructure.
The success path is otherwise the frozen StressProxy.call implementation.
"""
import copy
import time
from urllib.parse import parse_qs, urlsplit

from medtoolbench.research.transport import Proxy
from .corrected_runner import InterfaceConsistentProxy

REVISION = 'concurrent-client-rejection-v1'


class ConcurrentRejectionProxy(InterfaceConsistentProxy):
    def call(self, method, path, body=None, actor='agent'):
        started = time.perf_counter()
        method = method.upper()
        total = self.triggered
        chosen = 'clean'
        repeats = int(self.spec.get('repeat', 1))
        for h in self.components:
            if h == 'clean':
                continue
            if h != 'interface_drift' and self.counts[h] >= repeats:
                continue
            if h == 'transient_503' and self.spec.get('placement') == 'agent_target' and (actor != 'agent' or not self.target(method, path, body)):
                continue
            if h in {'stale', 'patient_mismatch', 'provenance_conflict'} and (method != 'GET' or not self.target(method, path, body) or path.startswith('/authoritative/')):
                continue
            if h in {'commit_timeout', 'no_commit_timeout'} and (method != 'POST' or not self.target(method, path, body)):
                continue
            if h == 'interface_drift' and not (method == 'GET' and 'patient' in parse_qs(urlsplit(path).query) and 'Observation' in path):
                continue
            chosen = h
            break
        other = None
        if chosen in {'commit_timeout', 'no_commit_timeout'} and self.spec.get('concurrent') and not self.concurrent_done:
            other = self.store.call(method, path, body)
            if 400 <= other['status'] < 500:
                self.trace.append({'method': method, 'path': path, 'body': copy.deepcopy(body),
                    'actor': actor, 'response': copy.deepcopy(other), 'injected': False,
                    'committed_response': None, 'fault_component': None,
                    'concurrent_write_rejected': True, 'wall_seconds': time.perf_counter() - started})
                return copy.deepcopy(other)
            if other['status'] >= 400:
                raise RuntimeError('Controlled concurrent write failed: ' + str(other))
            self.concurrent_done = True
            self.external_ids.add(other['body']['id'])
        self.hazard, self.triggered = chosen, 0
        result = Proxy.call(self, method, path, body, actor)
        fired = bool(self.trace[-1]['injected'])
        if fired:
            self.counts[chosen] += 1
        self.triggered = total + int(fired)
        record = self.trace[-1]
        record['fault_component'] = chosen if fired else None
        if other:
            record['external_committed_response'] = other
            if not self.mutation_states or self.mutation_states[-1]['trace_index'] != len(self.trace) - 1:
                self.mutation_states.append({'trace_index': len(self.trace) - 1, 'state': self.store.snapshot()})
        committed = record.get('committed_response')
        if committed and committed['status'] < 400:
            self.hidden_ids.add(committed['body']['id'])
        if method == 'GET' and self.delay_left and self.hidden_ids and result['body'].get('resourceType') == 'Bundle':
            entries = result['body'].get('entry', [])
            if any(e.get('resource', {}).get('id') in self.hidden_ids for e in entries):
                result = copy.deepcopy(result)
                result['body']['entry'] = [e for e in entries if e.get('resource', {}).get('id') not in self.hidden_ids]
                result['body']['total'] = len(result['body']['entry'])
                self.delay_left -= 1
                record['response'] = copy.deepcopy(result)
                record['visibility_masked'] = True
        return result
