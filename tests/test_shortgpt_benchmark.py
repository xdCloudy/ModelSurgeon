"""Tests for the reproducible ShortGPT physical benchmark report."""

from tools.benchmark_shortgpt_depth import _limitations


def test_report_limitations_reflect_the_number_of_random_seeds() -> None:
    limitations = _limitations(5)

    assert "5 predeclared random seeds give descriptive, not powered, uncertainty estimates." in (
        limitations
    )
    assert all("Three predeclared random seeds" not in item for item in limitations)
