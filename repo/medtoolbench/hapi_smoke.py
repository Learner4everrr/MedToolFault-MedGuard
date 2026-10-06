"""Validate write-uncertainty mechanisms against the author's real HAPI binary.

Starts its OWN loopback-only server with a fresh temporary database, never uses
the preloaded upstream patient database and never connects to external EHRs.
"""
import argparse
import copy
import json
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .construction import suite
from .grading import grade
from .hazards import Proxy
from .oracle import solve
from .storage import digest, outcome

KINDS = ("Patient", "Observation", "MedicationRequest", "ServiceRequest")


class OwnedHapiStore:
    def __init__(self, base):
        assert urlsplit(base).hostname == "127.0.0.1"
        self.base = base

    def call(self, method, path, body=None):
        if not path.startswith("/") or urlsplit(path).netloc or urlsplit(path).scheme or ".." in path:
            return outcome(400, "Only local sandbox paths allowed")
        request = Request(self.base + path, method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Accept": "application/fhir+json", "Content-Type": "application/fhir+json",
                     "Cache-Control": "no-cache"})
        try:
            with urlopen(request, timeout=60) as response:
                raw = response.read()
                return {"status": response.status, "body": json.loads(raw) if raw else {}}
        except HTTPError as e:
            raw = e.read()
            return {"status": e.code, "body": json.loads(raw) if raw else {}}

    def snapshot(self):
        resources = []
        for kind in KINDS:
            result = self.call("GET", f"/{kind}?_count=200&_total=accurate")
            if result["status"] != 200:
                raise RuntimeError(result)
            b = result["body"]
            if any(x["relation"] == "next" for x in b.get("link", [])):
                raise RuntimeError("Smoke fixture unexpectedly exceeds one page")
            resources.extend(e["resource"] for e in b.get("entry", []))
        return sorted(resources, key=lambda r: (r["resourceType"], r["id"]))

    def reload(self, resources):
        current = self.snapshot()
        # Deletion is only against this process-owned, freshly created sandbox DB.
        # Child resources precede patients to respect reference integrity.
        for r in sorted(current, key=lambda r: r["resourceType"] == "Patient"):
            result = self.call("DELETE", f"/{r['resourceType']}/{r['id']}")
            if result["status"] >= 400:
                raise RuntimeError(result)
        assert not self.snapshot()
        for r in sorted(resources, key=lambda r: r["resourceType"] != "Patient"):
            result = self.call("PUT", f"/{r['resourceType']}/{r['id']}", r)
            if result["status"] >= 400:
                raise RuntimeError(result)
        # Server-generated ids/meta on PUT are not clinically meaningful changes.
        from .grading import canonical
        assert [canonical(r) for r in self.snapshot()] == [canonical(r) for r in sorted(resources, key=lambda r: (r["resourceType"], r["id"]))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--war", type=Path, required=True)
    p.add_argument("--image", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise SystemExit("Refusing to overwrite existing HAPI validation output")
    args.out.mkdir(parents=True)
    work = tempfile.TemporaryDirectory(prefix="medtoolbench-hapi-")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    root = Path(work.name)
    java_prefix = ["apptainer", "exec", "--no-home", str(args.image.resolve()), "java"] if args.image else ["java"]
    classpath = "/app/main.war" if args.image else str(args.war.resolve())
    command = java_prefix + ["-Xmx3g", "-cp", classpath,
        "org.springframework.boot.loader.WarLauncher", "--server.address=127.0.0.1",
        f"--server.port={port}", f"--spring.datasource.url=jdbc:h2:file:{root}/isolated_db",
        "--hapi.fhir.reuse_cached_search_results_millis=0", "--hapi.fhir.narrative_enabled=false"]
    (args.out / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    rows = []
    proc = None
    with (args.out / "server.log").open("w") as log:
        try:
            proc = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            store = OwnedHapiStore(f"http://127.0.0.1:{port}/fhir")
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError("HAPI startup failed; inspect server.log")
                try:
                    metadata = store.call("GET", "/metadata")
                    if metadata["status"] == 200:
                        break
                except (URLError, TimeoutError, ConnectionError):
                    pass
                time.sleep(1)
            else:
                raise TimeoutError("HAPI did not start within 180 seconds")
            (args.out / "capability.json").write_text(json.dumps(metadata["body"], indent=2) + "\n")
            assert not store.snapshot(), "Refusing to use a nonempty EHR server"
            for task in suite(4):
                for hazard in task["hazards"]:
                    for condition in ("medguard", "retry_only"):
                        store.reload(task["resources"])
                        before = store.snapshot()
                        proxy = Proxy(store, task, hazard)
                        answer = solve(task, proxy, condition)
                        after = store.snapshot()
                        g = grade(task, before, after, answer)
                        rows.append({"task_id": task["id"], "family": task["family"], "hazard": hazard,
                            "condition": condition, "grade": g, "before": before, "after": after,
                            "trace": proxy.trace, "triggered": proxy.triggered})
                        if condition == "medguard":
                            assert g["success"] and proxy.triggered == int(hazard != "clean"), rows[-1]
                        if condition == "retry_only" and hazard == "commit_timeout":
                            assert "duplicate" in g["unsafe"], rows[-1]
                        print(task["id"], hazard, condition, g["success"], flush=True)
            (args.out / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            (args.out / "complete.json").write_text(json.dumps({"episodes": len(rows),
                "medguard_oracle_passed": sum(r["condition"] == "medguard" and r["grade"]["success"] for r in rows),
                "retry_duplicates": sum("duplicate" in r["grade"]["unsafe"] for r in rows),
                "backend": "Official MedAgentBench image HAPI binary; fresh synthetic database",
                "real_mab_patient_data": False, "model_inference": False}, indent=2) + "\n")
        finally:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
            # Archive only this run's DB before cleaning its temporary directory.
            for f in root.glob("isolated_db*"):
                if f.is_file():
                    shutil.copy2(f, args.out / f.name)
            work.cleanup()


if __name__ == "__main__":
    main()
