"""Read-only probe of an isolated COPY of the official image's preloaded EHR.

This prepares evidence about authentic-substrate availability. It is NOT a model
evaluation or benchmark conversion; no patient record is changed through FHIR.
"""
import argparse
import json
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.error import URLError

from .hapi_smoke import OwnedHapiStore


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise SystemExit("Refusing to overwrite prior substrate probe")
    args.out.mkdir(parents=True)
    tasks = json.loads((args.upstream / "MedAgentBench/data/medagentbench/test_data_v1.json").read_text())
    tasks = [t for t in tasks if t["id"].split("_")[0] in {"task4", "task5", "task7"}]
    temp = tempfile.TemporaryDirectory(prefix="medtoolbench-upstream-probe-")
    root = Path(temp.name)
    source = args.upstream / "fhir_root/data/test_db.mv.db"
    shutil.copy2(source, root / "source_db.mv.db")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    command = ["apptainer", "exec", "--no-home", str((args.upstream / "medagentbench.sif").resolve()),
        "java", "-Xmx3g", "-cp", "/app/main.war", "org.springframework.boot.loader.WarLauncher",
        "--server.address=127.0.0.1", f"--server.port={port}",
        f"--spring.datasource.url=jdbc:h2:file:{root}/source_db", "--hapi.fhir.narrative_enabled=false"]
    proc = None
    rows = []
    with (args.out / "server.log").open("w") as log:
        try:
            proc = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            store = OwnedHapiStore(f"http://127.0.0.1:{port}/fhir")
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError("Preloaded HAPI startup failed; inspect server.log")
                try:
                    if store.call("GET", "/metadata")["status"] == 200:
                        break
                except (URLError, TimeoutError, ConnectionError):
                    pass
                time.sleep(1)
            else:
                raise TimeoutError("Preloaded HAPI startup timed out")
            summary = store.call("GET", "/Patient?_summary=count")
            for task in tasks:
                code = "GLU" if task["id"].startswith("task7_") else "MG"
                path = f"/Observation?patient={task['eval_MRN']}&code={code}&_count=200"
                response = store.call("GET", path)
                rows.append({"source_task_id": task["id"], "source_task": task, "query": path,
                             "response": response})
                print(task["id"], response["status"], len(response["body"].get("entry", [])), flush=True)
            (args.out / "read_samples.json").write_text(json.dumps(rows, indent=2) + "\n")
            report = {"source": "official MedAgentBench preloaded database; copied before opening",
                "patient_count_response": summary, "tasks_probed": len(rows),
                "nonempty_observation_responses": sum(bool(r["response"]["body"].get("entry")) for r in rows),
                "responses_with_pagination": sum(any(x["relation"] == "next" for x in r["response"]["body"].get("link", [])) for r in rows),
                "fhir_mutations": 0, "model_inference": False,
                "warning": "Probe only. Not a corrected task manifest or MAB-v3 reproduction. Do not redistribute without data-license review."}
            (args.out / "complete.json").write_text(json.dumps(report, indent=2) + "\n")
        finally:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
            temp.cleanup()


if __name__ == "__main__":
    main()
