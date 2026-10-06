"""Use the explicitly versioned V2 metadata semantics in full oracle replays."""
import argparse
import sys

from .corrected_runner import InterfaceConsistentProxy

p = argparse.ArgumentParser()
p.add_argument("--kind", choices=["primary", "actions"], required=True)
a, rest = p.parse_known_args()
if a.kind == "primary":
    from . import oracle as module
else:
    from . import action_oracle as module
module.StressProxy = InterfaceConsistentProxy
sys.argv = [module.__file__, *rest]
module.main()
