"""Restricted, resource-limited arithmetic over already-retrieved FHIR objects.

No model-generated file/network/process access. This is a constrained Python
subset, not a general-purpose secure Python execution service.
"""
import ast
import contextlib
import datetime
import io
import json
import math
import resource
import statistics
import sys
import time


def internal_import(name, globals=None, locals=None, fromlist=(), level=0):
    # datetime.strftime imports time internally. Model code cannot spell any
    # private identifier, and all source import nodes are separately checked.
    if name == "time" and not level:
        return time
    raise ImportError("Internal import unavailable")

ATTRS = {"get", "items", "keys", "values", "setdefault", "append", "extend", "sort", "copy",
    "lower", "upper", "strip", "lstrip", "rstrip", "split", "rsplit", "replace", "startswith", "endswith",
    "join", "find", "count", "casefold", "removeprefix", "removesuffix", "isdigit",
    "datetime", "date", "timedelta", "timezone", "utc", "fromisoformat", "isoformat", "strftime",
    "year", "month", "day", "hour", "minute", "second", "days", "seconds", "total_seconds",
    "mean", "median", "fmean", "isclose", "isfinite", "floor", "ceil", "sqrt"}
BUILTINS = {"len": len, "sum": sum, "min": min, "max": max, "sorted": sorted, "range": range,
    "enumerate": enumerate, "zip": zip, "list": list, "dict": dict, "set": set, "tuple": tuple,
    "str": str, "int": int, "float": float, "bool": bool, "abs": abs, "round": round,
    "any": any, "all": all, "isinstance": isinstance, "reversed": reversed,
    "print": print, "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
    "KeyError": KeyError, "IndexError": IndexError, "__import__": internal_import}
MODULES = {"datetime": datetime, "math": math, "statistics": statistics}


def checked_tree(code):
    if not isinstance(code, str) or len(code) > 24000:
        raise ValueError("Code must be at most 24,000 characters")
    tree = ast.parse(code)
    bindings = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef, ast.With, ast.AsyncWith,
                             ast.Global, ast.Nonlocal, ast.Delete, ast.Await, ast.Yield, ast.YieldFrom)):
            raise ValueError("Unsupported Python statement")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise ValueError("Private names are unavailable")
        if isinstance(node, ast.Attribute) and node.attr not in ATTRS:
            raise ValueError("Unsupported attribute: " + node.attr)
        if isinstance(node, ast.Pow):
            raise ValueError("Exponentiation unavailable")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in MODULES or (alias.asname or alias.name).startswith("_"):
                    raise ValueError("Only datetime, math and statistics imports are supported")
                bindings[alias.asname or alias.name] = MODULES[alias.name]
        if isinstance(node, ast.ImportFrom):
            if node.module not in MODULES or node.level:
                raise ValueError("Unsupported import")
            for alias in node.names:
                if alias.name not in ATTRS or (alias.asname or alias.name).startswith("_"):
                    raise ValueError("Unsupported imported name")
                bindings[alias.asname or alias.name] = getattr(MODULES[node.module], alias.name)
    class RemoveImports(ast.NodeTransformer):
        def visit_Import(self, node):
            return ast.copy_location(ast.Pass(), node)
        def visit_ImportFrom(self, node):
            return ast.copy_location(ast.Pass(), node)
    return ast.fix_missing_locations(RemoveImports().visit(tree)), bindings


class LimitedOutput(io.StringIO):
    def write(self, text):
        if self.tell() + len(text) > 24000:
            raise ValueError("Code output exceeds 24,000 characters")
        return super().write(text)


def evaluate(code, retrieved, verification=None):
    tree, bindings = checked_tree(code)
    scope = {"__builtins__": BUILTINS, "retrieved_resources": retrieved,
        "verification_resources": verification or {}, "datetime": datetime, "math": math,
        "statistics": statistics, **bindings}
    stream = LimitedOutput()
    with contextlib.redirect_stdout(stream):
        exec(compile(tree, "<restricted-fhir-calculation>", "exec"), scope, scope)
    answer = {"stdout": stream.getvalue()}
    if "result" in scope:
        value = scope["result"]
        encoded = json.dumps(value, allow_nan=False)
        if len(encoded) > 24000:
            raise ValueError("Result is too large")
        answer["result"] = value
    return answer


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (8, 8))
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    try:
        request = json.load(sys.stdin)
        response = evaluate(request["code"], request["retrieved_resources"], request.get("verification_resources"))
        print(json.dumps({"ok": True, **response}, allow_nan=False))
    except BaseException as error:
        print(json.dumps({"ok": False, "error": type(error).__name__ + ": " + str(error)[:1000]}))


if __name__ == "__main__":
    main()
