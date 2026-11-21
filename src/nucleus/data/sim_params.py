import json
from pathlib import Path


def load_sim_params(filename: str) -> dict:
    sidecar = Path(filename).with_suffix(".json")
    with sidecar.open(encoding="utf-8") as handle:
        sim_params = json.load(handle)

    if sidecar.parent.name == "SingleBubble-Saturated-FC72-2D":
        sim_params.setdefault("setup", "saturated")
        sim_params.setdefault("liquid", "fc72")
        # Both temperatures are 58 °C in the single-bubble divfree config.
        sim_params.setdefault("sat_temp", 58)
        sim_params.setdefault("bulk_temp", sim_params["sat_temp"])

    return sim_params
