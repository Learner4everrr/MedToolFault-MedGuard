"""C4 execution: unchanged agent/fault/grader with corrected fixture metadata."""
from .corrected_runner import InterfaceConsistentProxy
from .profile_store import ProfilePreservingProcess


def main():
    from . import runner
    runner.HapiProcess = ProfilePreservingProcess
    runner.StressProxy = InterfaceConsistentProxy
    runner.main()


if __name__ == "__main__":
    main()
