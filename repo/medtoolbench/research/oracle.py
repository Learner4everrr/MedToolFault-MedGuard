"""Online scripted recovery oracle: receives public instructions, not gold state."""
from datetime import date, timedelta
from urllib.parse import urlencode

from .contracts import CLOCK, NOW, READ_CODES, concept, instant, medication, quantity, service
from .transport import Client


def public_task(task):
    return {k: task[k] for k in ("template", "patient", "parameters", "prompt")}


def solve(task, call, condition="medguard"):
    """Independent online workflow, using only returned tool results."""
    client = Client(call, condition)
    t, p, params = task["template"], task["patient"], task["parameters"]

    def get(path):
        result = client.call("GET", path)
        if result["status"] != 200:
            raise RuntimeError(result)
        return result["body"]

    def all_rows(path):
        rows, seen = [], set()
        while path:
            if path in seen:
                raise RuntimeError("Oracle pagination cycle")
            seen.add(path)
            b = get(path)
            rows.extend(e["resource"] for e in b.get("entry", []) if e.get("search", {}).get("mode", "match") == "match")
            path = next((x["url"] for x in b.get("link", []) if x["relation"] == "next"), None)
        return rows

    def create(resource):
        r = client.call("POST", "/" + resource["resourceType"], resource)
        if r["status"] >= 400:
            raise RuntimeError(r)

    if t == 1:
        # FHIR name search matches name parts, not necessarily a concatenated
        # full-name string. Apply exact full-name matching after a DOB search.
        rows = all_rows("/Patient?" + urlencode({"birthdate": params["birthDate"], "_count": 2000}))
        exact = [r for r in rows if any(" ".join(n.get("given", []) + [n.get("family", "")]).casefold() == params["name"].casefold()
                                       for n in r.get("name", [])) and r.get("birthDate") == params["birthDate"]]
        return {"mrn": exact[0]["id"] if exact else "Patient not found"}
    if t == 2:
        birth = date.fromisoformat(get(f"/Patient/{p}")["birthDate"])
        today = CLOCK.date()
        years = today.year - birth.year
        if (birth.month, birth.day) > (today.month, today.day):
            years -= 1
        return {"age": years}
    if t == 3:
        create({"resourceType": "Observation", "status": "final", "code": {"text": "BP"},
                "subject": {"reference": f"Patient/{p}"}, "effectiveDateTime": NOW,
                "component": [{"code": concept("http://loinc.org", c), "valueQuantity": {"value": n, "unit": "mm[Hg]"}}
                              for c, n in [("8480-6", params["bp"][0]), ("8462-4", params["bp"][1])]]})
        return {"completed": True}
    if t == 8:
        create(service(p, "306181000000106", "http://snomed.info/sct", note=params["note"]))
        return {"completed": True}
    code = READ_CODES[t]
    query = [("patient", p), ("code", code), ("status", "final,amended,corrected"),
             ("date", "le" + NOW), ("_sort", "-date"), ("_count", "2000" if t == 6 else "1")]
    if t in {4, 5, 6}:
        query.append(("date", "ge" + (CLOCK - timedelta(days=1)).isoformat()))
    if t == 6:
        rows = all_rows("/Observation?" + urlencode(query))
    else:
        b = get("/Observation?" + urlencode(query))
        rows = [e["resource"] for e in b.get("entry", [])]
    value = quantity(rows[0], code) if rows else -1
    if t in {4, 7}:
        return {"value": value}
    if t == 6:
        return {"value": sum(quantity(r, code) for r in rows) / len(rows) if rows else -1}
    if t == 5:
        if rows and value < 1:
            create(medication(p, "0338-1715-40", 4, "g", "Intravenous", 4))
        elif rows and value < 1.5:
            create(medication(p, "0338-1715-40", 2, "g", "Intravenous", 2))
        elif rows and value <= 1.9:
            create(medication(p, "0338-1715-40", 1, "g", "Intravenous", 1))
        return {"completed": True}
    if t == 9:
        if rows and value < 3.5:
            dose = round(10 * ((3.5 - value) / .1), 6)
            create(medication(p, "40032-917-01", dose, "mEq", "Oral"))
            create(service(p, "2823-3", when="2023-11-14T08:00:00+00:00"))
        return {"completed": True}
    if t == 10:
        if rows and instant(rows[0]["effectiveDateTime"]) < CLOCK.replace(year=2022):
            create(service(p, "4548-4"))
        return {"value": value, "date": rows[0]["effectiveDateTime"][:10] if rows else None}
    raise ValueError(t)
