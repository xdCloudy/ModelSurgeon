import pytest

from modelsurgeon.evaluation import (
    DEFAULT_ARCHITECTURE_PARETO_STUDY,
    ArchitectureFrontierPoint,
    ArchitectureParetoStudyConfig,
    ArchitectureParetoStudyError,
    ArchitectureStudyCell,
    ArchitectureStudyKey,
    ArchitectureStudyMetric,
    ArchitectureStudyOutcome,
    build_architecture_pareto_study,
    compute_frontier,
    compute_hypervolume,
)


def _point(
    state: str, quality: float, cost: float, family: str, hardware: str
) -> ArchitectureFrontierPoint:
    return ArchitectureFrontierPoint(
        state,
        "a" * 64,
        family,
        hardware,
        quality,
        quality,
        quality,
        cost,
        cost,
        cost,
        3,
        (("artifact", "fixture-artifact-v1"), ("runtime", "fixture-runtime-v1")),
    )


def _config() -> ArchitectureParetoStudyConfig:
    return ArchitectureParetoStudyConfig(
        ("family-a", "family-b"),
        ("cpu", "gpu"),
        ("beam", "greedy"),
        (1, 2),
        (11, 23, 47),
        bootstrap_repetitions=100,
    )


def _cells(config: ArchitectureParetoStudyConfig) -> tuple[ArchitectureStudyCell, ...]:
    cells = []
    for family in config.families:
        for hardware in config.hardware_profiles:
            for method in config.methods:
                for budget in config.budgets:
                    for seed in config.seeds:
                        quality = 0.9 if method == "beam" else 0.8
                        point = _point(
                            f"state_{family}_{hardware}_{method}_{budget}_{seed}",
                            quality,
                            2.0 if method == "beam" else 3.0,
                            family,
                            hardware,
                        )
                        cells.append(
                            ArchitectureStudyCell(
                                ArchitectureStudyKey(family, hardware, method, budget, seed),
                                ArchitectureStudyOutcome.MEASURED,
                                (point,),
                                (
                                    ArchitectureStudyMetric(
                                        "hypervolume",
                                        2.0 if method == "beam" else 1.0,
                                        2.0 if method == "beam" else 1.0,
                                        2.0 if method == "beam" else 1.0,
                                        3,
                                        "fixture-hypervolume-v1",
                                    ),
                                ),
                            )
                        )
    return tuple(cells)


def test_default_study_retains_unsupported_matrix_without_claim() -> None:
    study = DEFAULT_ARCHITECTURE_PARETO_STUDY
    assert len(study.cells) == 2 * 2 * 6 * 2 * 3
    assert {cell.outcome for cell in study.cells} == {ArchitectureStudyOutcome.UNSUPPORTED}
    assert study.decision.value == "unsupported"
    assert study.comparisons


def test_study_reports_paired_positive_interval_and_frontier_area() -> None:
    study = build_architecture_pareto_study(_config(), _cells(_config()))
    assert study.decision.value == "positive"
    assert all(
        comparison.lower is not None and comparison.lower > 0 for comparison in study.comparisons
    )
    points = (
        _point("state_a", 0.8, 3, "family-a", "cpu"),
        _point("state_b", 0.9, 2, "family-a", "cpu"),
        _point("state_c", 0.7, 4, "family-a", "cpu"),
    )
    frontier = compute_frontier(points)
    assert points[1].point_id in frontier
    assert points[2].point_id not in frontier
    assert compute_hypervolume(points) >= 0


def test_study_retains_negative_cells_and_rejects_incomplete_matrix() -> None:
    config = _config()
    cells = list(_cells(config))
    negative = cells[0]
    cells[0] = ArchitectureStudyCell(
        negative.key,
        ArchitectureStudyOutcome.NEGATIVE_RESULT,
        negative.frontier_points,
        negative.metrics,
        "beam did not improve the held-out frontier",
    )
    assert build_architecture_pareto_study(config, tuple(cells)).cells[0].reason
    with pytest.raises(ArchitectureParetoStudyError, match="complete comparison matrix"):
        build_architecture_pareto_study(config, tuple(cells[:-1]))
    with pytest.raises(ArchitectureParetoStudyError, match="measured cells require"):
        ArchitectureStudyCell(
            negative.key,
            ArchitectureStudyOutcome.MEASURED,
            (),
            (),
        )
