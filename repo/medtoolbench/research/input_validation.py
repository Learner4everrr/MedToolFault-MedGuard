"""Validate untrusted tool syntax without silently correcting clinical intent."""
import json
import re
from urllib.parse import unquote, urlsplit


def validate_path(path):
    if not isinstance(path, str) or not path or len(path) > 16384:
        raise ValueError("Tool path must be a nonempty string of at most 16384 characters")
    if any(ord(c) <= 32 or ord(c) >= 127 for c in path):
        raise ValueError("URL path contains whitespace/control/non-ASCII characters; percent-encode query values (space=%20)")
    if re.search(r"%(?![0-9a-fA-F]{2})", path):
        raise ValueError("Malformed percent encoding in tool path")
    parsed = urlsplit(path)
    if parsed.scheme or parsed.netloc or parsed.fragment or not path.startswith("/") or ".." in unquote(parsed.path):
        raise ValueError("Only owned sandbox relative paths are permitted")
    if any(ord(c) < 32 or ord(c) == 127 for c in unquote(path)):
        raise ValueError("Encoded control characters are not permitted")
    return parsed


def validate_body(body):
    if body is None:
        return
    if not isinstance(body, dict):
        raise ValueError("FHIR body must be a JSON object")
    json.dumps(body, allow_nan=False)
    if "resourceType" in body and not isinstance(body["resourceType"], str):
        raise ValueError("resourceType must be a string")
    if "subject" in body:
        ref = body["subject"]
        if not isinstance(ref, dict) or ("reference" in ref and not isinstance(ref["reference"], str)):
            raise ValueError("subject must be an object with a string reference")
