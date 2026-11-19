import json

import pytest

from nucleus.training.data import read_split


def _split_dir(tmp_path, manifest):
    for split, cases in manifest.items():
        for case in cases:
            path = tmp_path / split / case
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    (tmp_path / "split.json").write_text(json.dumps({"seed": 0, **manifest}))
    return tmp_path


def test_read_split_finds_each_case_under_its_split(tmp_path):
    split_dir = _split_dir(tmp_path, {
        "train": ["FC72/Twall_100.hdf5", "R515B/Twall_100.hdf5"],
        "val": ["FC72/Twall_90.hdf5"],
        "test": [],
    })
    split = read_split(split_dir)
    assert split["train"] == [split_dir / "train/FC72/Twall_100.hdf5", split_dir / "train/R515B/Twall_100.hdf5"]
    assert split["val"] == [split_dir / "val/FC72/Twall_90.hdf5"]
    assert split["test"] == []


def test_read_split_reports_cases_missing_from_the_directory(tmp_path):
    split_dir = _split_dir(tmp_path, {"train": ["a.hdf5"], "val": [], "test": []})
    manifest = json.loads((split_dir / "split.json").read_text())
    manifest["test"] = ["gone.hdf5"]
    (split_dir / "split.json").write_text(json.dumps(manifest))
    with pytest.raises(FileNotFoundError, match="gone.hdf5"):
        read_split(split_dir)
