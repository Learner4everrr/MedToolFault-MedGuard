"""Full C4 public-interface oracle; same tasks, solver and strict state checks."""
from .corrected_runner import InterfaceConsistentProxy
from .profile_store import ProfilePreservingProcess


def main():
    from . import action_oracle
    action_oracle.HapiProcess = ProfilePreservingProcess
    action_oracle.StressProxy = InterfaceConsistentProxy
    action_oracle.main()


if __name__ == "__main__":
    main()
