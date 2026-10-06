"""C-only fixture import correction; preserve non-server-owned FHIR metadata."""
import copy

from medtoolbench.storage import digest
from medtoolbench.research.fhir import HapiProcess, HapiStore, canonical, normalized_state, state_key


def import_payload(resource):
    resource = copy.deepcopy(resource)
    if "meta" in resource:
        for name in ("versionId", "lastUpdated", "source"):
            resource["meta"].pop(name, None)
        if not resource["meta"]:
            resource.pop("meta")
    return resource


class ProfilePreservingStore(HapiStore):
    def reload(self, resources):
        # Keep the original transaction/reset/check semantics. The sole payload
        # change is retaining profile/security/tag rather than dropping all meta.
        if self.owner.database_copy:
            raise RuntimeError("Refusing writes to a source database probe, even though it is a copy")
        fixture = sorted(copy.deepcopy(resources), key=state_key)
        desired = {state_key(r): r for r in fixture}
        current = {state_key(r): r for r in self.snapshot()}
        entries = []
        for key in sorted(current, key=lambda k: k[0] == "Patient"):
            if key not in desired:
                entries.append({"request": {"method": "DELETE", "url": "/".join(key)}})
        for key in sorted(desired, key=lambda k: k[0] != "Patient"):
            if key not in current or canonical(current[key]) != canonical(desired[key]):
                entries.append({"request": {"method": "PUT", "url": "/".join(key)}, "resource": import_payload(desired[key])})
        self.transaction(entries)
        after = self.snapshot()
        if normalized_state(after) != normalized_state(fixture):
            raise AssertionError("HAPI reset state differs from declared fixture")
        self.fixture = fixture
        self.fixture_hash = digest(normalized_state(fixture))
        return after


class ProfilePreservingProcess(HapiProcess):
    def start(self):
        super().start()
        # The initial capability probe does not load a fixture. Replace only
        # this owned instance's store, never the frozen shared module globals.
        self.store = ProfilePreservingStore(self)
        return self.store
