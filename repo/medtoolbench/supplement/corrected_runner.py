"""V2 fault metadata correction; original V1 sources/evidence remain unchanged."""
from .transport import StressProxy


class InterfaceConsistentProxy(StressProxy):
    def metadata(self):
        response = super().metadata()
        # StressProxy routes non-faulted operations with self.hazard='clean'.
        # Active interface metadata must instead reflect the declared component,
        # even when this particular metadata read does not itself trigger a fault.
        if "interface_drift" in self.components:
            for rest in response["body"].get("rest", []):
                for resource in rest.get("resource", []):
                    if resource.get("type") == "Observation":
                        resource["searchParam"] = [p for p in resource.get("searchParam", []) if p["name"] != "patient"]
        return response


def main():
    from . import runner
    runner.StressProxy = InterfaceConsistentProxy
    runner.main()


if __name__ == "__main__":
    main()
