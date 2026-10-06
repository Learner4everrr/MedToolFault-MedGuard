"""Process-owned local HAPI lifecycle, paginated reads and validated state reset."""
from __future__ import annotations

import copy
import json
import shutil
import socket
import subprocess
import tempfile
import time
from http.client import InvalidURL
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, unquote, urlsplit
from urllib.request import Request, urlopen

from .contracts import state_payload as canonical
from medtoolbench.storage import digest, outcome
from .input_validation import validate_body, validate_path

PRIMARY_TYPES = ("Patient", "Observation", "MedicationRequest", "ServiceRequest")


def state_key(r):
    return r["resourceType"], r["id"]


def normalized_state(resources):
    return [{"key": list(state_key(r)), "clinical": canonical(r)} for r in sorted(resources, key=state_key)]


class HapiProcess:
    def __init__(self, image, logs, database_copy=None, heap="1536m"):
        self.image = Path(image).resolve()
        self.logs = Path(logs)
        self.database_copy = database_copy
        self.heap = heap
        self.temp = None
        self.proc = None
        self.handle = None

    def start(self):
        self.logs.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="medtoolbench-hapi-research-")
        root = Path(self.temp.name)
        if self.database_copy:
            shutil.copy2(self.database_copy, root / "state.mv.db")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        command = ["apptainer", "exec", "--no-home", str(self.image), "java", "-Xmx" + self.heap,
            "-XX:ActiveProcessorCount=2", "-cp", "/app/main.war", "org.springframework.boot.loader.WarLauncher",
            "--server.address=127.0.0.1", f"--server.port={self.port}",
            f"--spring.datasource.url=jdbc:h2:file:{root}/state",
            "--hapi.fhir.narrative_enabled=false", "--hapi.fhir.reuse_cached_search_results_millis=0",
            # The source fixture is an explicitly declared resource subset; preserve original
            # external encounter/provider references without inventing placeholder patients.
            "--hapi.fhir.enforce_referential_integrity_on_write=false",
            "--hapi.fhir.enforce_referential_integrity_on_delete=false",
            "--hapi.fhir.default_page_size=200", "--hapi.fhir.max_page_size=2000",
            "--logging.level.root=WARN"]
        (self.logs / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        self.handle = (self.logs / "server.log").open("w")
        self.proc = subprocess.Popen(command, cwd=root, stdout=self.handle, stderr=subprocess.STDOUT)
        self.store = HapiStore(self)
        try:
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"HAPI exited at startup: {self.logs / 'server.log'}")
                try:
                    response = self.store.request("GET", "/metadata", timeout=3)
                    if response["status"] == 200:
                        (self.logs / "capability.json").write_text(json.dumps(response["body"], indent=2) + "\n")
                        return self.store
                except (URLError, TimeoutError, ConnectionError):
                    pass
                time.sleep(.5)
            raise TimeoutError("HAPI startup timed out")
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        if self.handle:
            self.handle.close()
        if self.temp:
            self.temp.cleanup()
            self.temp = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()


class HapiStore:
    """Never instantiate with arbitrary EHR URLs: only a process created above."""
    def __init__(self, owner):
        self.owner = owner
        self.base = f"http://127.0.0.1:{owner.port}/fhir"
        self.types = PRIMARY_TYPES
        self.fixture = None
        self.fixture_hash = None

    def request(self, method, path, body=None, timeout=90, headers=None):
        try:
            validate_path(path)
            validate_body(body)
        except (ValueError, TypeError, RecursionError) as error:
            return outcome(400, str(error))
        h = {"Accept": "application/fhir+json", "Content-Type": "application/fhir+json", "Cache-Control": "no-cache"}
        h.update(headers or {})
        req = Request(self.base + path, method=method, data=json.dumps(body).encode() if body is not None else None, headers=h)
        try:
            with urlopen(req, timeout=timeout) as response:
                raw = response.read()
                return {"status": response.status, "body": json.loads(raw) if raw else {}}
        except (InvalidURL, UnicodeError) as error:
            return outcome(400, "Invalid encoded tool URL: " + str(error))
        except HTTPError as e:
            raw = e.read()
            try:
                b = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                b = {"error": raw.decode(errors="replace")[:1000]}
            return {"status": e.code, "body": b}

    def call(self, method, path, body=None):
        try:
            parsed = validate_path(path)
            validate_body(body)
        except (ValueError, TypeError, RecursionError) as error:
            return outcome(400, str(error))
        segments = parsed.path.strip("/").split("/")
        kind = segments[0]
        if kind not in (*self.types, "metadata", ""):
            return outcome(400, "Resource type outside this experiment's declared sandbox")
        if not kind and method.upper() == "GET" and "_getpages" in parse_qs(parsed.query):
            return self.request("GET", path)
        if not kind or method.upper() not in {"GET", "POST", "PUT", "DELETE"} or len(segments) > 2:
            return outcome(405, "Only individual sandbox resource operations are available to the agent")
        # Do not let agents change server settings, execute operations or arbitrary bundles.
        if "$" in path:
            return outcome(400, "Server operations are unavailable in this sandbox")
        if kind == "metadata" and method.upper() != "GET":
            return outcome(405, "Capability metadata is read-only")
        if body is not None and (not isinstance(body, dict) or body.get("resourceType") != kind):
            return outcome(400, "body.resourceType must match the resource path")
        return self.request(method.upper(), path, body)

    def all_pages(self, path):
        rows, visited = [], set()
        while path:
            if path in visited:
                raise RuntimeError("FHIR pagination loop")
            visited.add(path)
            result = self.request("GET", path)
            if result["status"] != 200:
                raise RuntimeError((path, result))
            b = result["body"]
            if b.get("resourceType") != "Bundle":
                return [b]
            rows.extend(e["resource"] for e in b.get("entry", []) if e.get("search", {}).get("mode", "match") != "outcome")
            next_url = next((e["url"] for e in b.get("link", []) if e["relation"] == "next"), None)
            if next_url:
                if not next_url.startswith(self.base + "/") and not next_url.startswith(self.base + "?"):
                    raise RuntimeError("Pagination attempted to leave owned server")
                path = next_url[len(self.base):]
                if path.startswith("?"):
                    path = "/" + path
            else:
                path = None
        return rows

    def snapshot(self):
        rows = []
        for kind in self.types:
            rows.extend(self.all_pages(f"/{kind}?_count=2000&_total=accurate"))
        return sorted(rows, key=state_key)

    def transaction(self, entries):
        for offset in range(0, len(entries), 200):
            result = self.request("POST", "/", {"resourceType": "Bundle", "type": "transaction", "entry": entries[offset:offset+200]})
            if result["status"] >= 400:
                raise RuntimeError(result)
            for e in result["body"].get("entry", []):
                if int(e["response"]["status"].split()[0]) >= 400:
                    raise RuntimeError(e)

    def reload(self, resources):
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
                r = copy.deepcopy(desired[key])
                r.pop("meta", None)  # Source clinical provenance retained separately, not forged server version.
                entries.append({"request": {"method": "PUT", "url": "/".join(key)}, "resource": r})
        self.transaction(entries)
        after = self.snapshot()
        if normalized_state(after) != normalized_state(fixture):
            raise AssertionError("HAPI reset state differs from declared fixture")
        self.fixture = fixture
        self.fixture_hash = digest(normalized_state(fixture))
        return after

    def reset(self):
        if self.fixture is None:
            raise RuntimeError("No fixture loaded")
        return self.reload(self.fixture)
