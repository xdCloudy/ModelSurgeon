"""Executable checks for the stable consumer workflow documentation."""

from __future__ import annotations

import json
from pathlib import Path

from modelsurgeon.optimization import available_hardware_profiles, available_quality_profiles

ROOT = Path(__file__).parents[1]


def test_quickstart_notebook_is_valid_and_uses_stable_plan_api() -> None:
    notebook = json.loads(
        (ROOT / "docs" / "notebooks" / "optimize_plan_quickstart.ipynb").read_text(
            encoding="utf-8"
        )
    )
    assert notebook["nbformat"] == 4
    source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "build_optimize_plan" in source
    assert "write_optimize_plan" in source
    assert "trust_remote_code" not in source


def test_documented_profile_names_match_the_public_api() -> None:
    profiles = (ROOT / "docs" / "user-guides" / "profiles.md").read_text(encoding="utf-8")
    for profile in available_hardware_profiles() + available_quality_profiles():
        assert f"`{profile}`" in profiles
