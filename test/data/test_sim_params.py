import json

import pytest

from nucleus.data.sim_params import load_sim_params


@pytest.mark.parametrize(
    ("case_name", "metadata", "expected_metadata"),
    [
        (
            "SingleBubble-Saturated-FC72-2D",
            {},
            {"bulk_temp": 58, "sat_temp": 58, "setup": "saturated", "liquid": "fc72"},
        ),
        (
            "SingleBubble-Saturated-FC72-2D",
            {"sat_temp": 57.5},
            {"bulk_temp": 57.5, "sat_temp": 57.5, "setup": "saturated", "liquid": "fc72"},
        ),
        (
            "SingleBubble-Saturated-FC72-2D",
            {"bulk_temp": 56, "sat_temp": 57, "setup": "custom", "liquid": "custom"},
            {"bulk_temp": 56, "sat_temp": 57, "setup": "custom", "liquid": "custom"},
        ),
        ("PoolBoiling-Saturated-FC72-2D", {}, {}),
    ],
)
def test_load_sim_params(tmp_path, case_name, metadata, expected_metadata):
    case = tmp_path / case_name
    case.mkdir()
    sidecar = case / "Twall_87.json"
    contents = json.dumps({**metadata, "heater": {"wallTemp": 87}})
    sidecar.write_text(contents, encoding="utf-8")

    assert load_sim_params(str(sidecar.with_suffix(".hdf5"))) == {
        **expected_metadata,
        "heater": {"wallTemp": 87},
    }
    assert sidecar.read_text(encoding="utf-8") == contents
