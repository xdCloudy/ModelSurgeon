import json
from pathlib import Path

from typer.testing import CliRunner

from modelsurgeon.cli.app import app
from modelsurgeon.search.resume import SearchResumeStore


def _candidate(name: str, quality: float, scale: float) -> dict[str, object]:
    return {
        "candidate_id": f"candidate_{name}",
        "candidate_state_id": f"state_{name}",
        "parent_state_id": "state_root",
        "parent_checkpoint_id": "checkpoint_root",
        "objective_observations": [
            {"metric": "quality", "value": quality, "baseline_value": 1.0},
            {"metric": "perplexity", "value": scale, "baseline_value": 1.0},
            {"metric": "parameter_count", "value": scale, "baseline_value": 1.0},
            {"metric": "latency", "value": scale, "baseline_value": 1.0},
            {"metric": "memory", "value": scale, "baseline_value": 1.0},
            {"metric": "disk_size", "value": scale, "baseline_value": 1.0},
        ],
        "constraint_observations": [
            {
                "metric": "quality_retention",
                "value": quality,
                "baseline": "immutable_source",
            },
            {
                "metric": "perplexity_delta",
                "value": 0.1,
                "baseline": "immutable_source",
            },
            {
                "metric": "latency_gain",
                "value": 0.1,
                "baseline": "immutable_source",
            },
            {"metric": "peak_ram", "value": 800, "baseline": "absolute"},
            {"metric": "peak_vram", "value": 700, "baseline": "absolute"},
            {"metric": "disk", "value": 600, "baseline": "absolute"},
        ],
        "reward_uncertainty": 0.01,
    }


def _write_config(path: Path) -> None:
    record = {
        "schema_version": 1,
        "source_checkpoint_id": "checkpoint_root",
        "source_state_id": "state_root",
        "accepted_checkpoint_ids": ["checkpoint_root"],
        "frontier_checkpoint_ids": ["checkpoint_root"],
        "constraints": {
            "min_quality_retention_ratio": 0.95,
            "max_perplexity_delta": 0.2,
            "min_latency_gain_ratio": 0.05,
            "max_ram_bytes": 1000,
            "max_vram_bytes": 1000,
            "max_disk_bytes": 1000,
        },
        "objectives": {
            "terms": [
                {
                    "metric": "quality",
                    "direction": "maximize",
                    "normalization": "baseline_ratio",
                },
                {
                    "metric": "perplexity",
                    "direction": "minimize",
                    "normalization": "baseline_ratio",
                },
                {
                    "metric": "parameter_count",
                    "direction": "minimize",
                    "normalization": "baseline_ratio",
                },
                {
                    "metric": "latency",
                    "direction": "minimize",
                    "normalization": "baseline_ratio",
                },
                {
                    "metric": "memory",
                    "direction": "minimize",
                    "normalization": "baseline_ratio",
                },
                {
                    "metric": "disk_size",
                    "direction": "minimize",
                    "normalization": "baseline_ratio",
                },
            ]
        },
        "policy": {
            "kind": "greedy",
            "evaluation_budget": 2,
            "beam_width": 1,
            "exploration_weight": 0.0,
            "seed": 7,
        },
        "candidates": [
            _candidate("best", 0.99, 0.5),
            _candidate("second", 0.98, 0.7),
            _candidate("unsafe", 0.80, 0.1),
        ],
    }
    path.write_text(json.dumps(record), encoding="utf-8")


def test_search_dry_run_prints_constraints_budget_and_initial_pool_without_state(
    tmp_path: Path,
) -> None:
    config = tmp_path / "search.json"
    state = tmp_path / "search.sqlite3"
    _write_config(config)

    result = CliRunner().invoke(app, ["search", str(config), "--dry-run"])

    assert result.exit_code == 0, result.output
    record = json.loads(result.stdout)
    assert record["record_type"] == "search_dry_run"
    assert len(record["constraints"]["constraints"]) == 6
    assert len(record["objectives"]["terms"]) == 6
    assert record["budget"]["evaluation_budget"] == 2
    assert len(record["initial_pool"]) == 3
    assert record["accepted_checkpoint_lineage"] == ["checkpoint_root"]
    assert record["state_written"] is False
    assert not state.exists()


def test_search_start_and_resume_reserve_distinct_candidates_atomically(tmp_path: Path) -> None:
    config = tmp_path / "search.json"
    state = tmp_path / "search.sqlite3"
    _write_config(config)
    runner = CliRunner()

    started = runner.invoke(app, ["search", str(config), "--state", str(state)])
    assert started.exit_code == 0, started.output
    first = json.loads(started.stdout)
    assert first["generation"] == 0
    assert first["resumed"] is False
    assert first["accepted_checkpoint_lineage"] == ["checkpoint_root"]
    assert [item["candidate_id"] for item in first["pending_evaluations"]] == ["candidate_best"]

    resumed = runner.invoke(
        app,
        ["search", str(config), "--state", str(state), "--resume"],
    )
    assert resumed.exit_code == 0, resumed.output
    second = json.loads(resumed.stdout)
    assert second["generation"] == 1
    assert second["resumed"] is True
    assert [item["candidate_id"] for item in second["pending_evaluations"]] == [
        "candidate_best",
        "candidate_second",
    ]
    with SearchResumeStore(state) as store:
        latest = store.load_latest(second["search_id"])
    assert latest.generation == 1
    assert latest.policy_state.selected_candidate_ids == (
        "candidate_best",
        "candidate_second",
    )
    assert latest.budget.evaluations_reserved == 2


def test_search_requires_state_for_nondry_run(tmp_path: Path) -> None:
    config = tmp_path / "search.json"
    _write_config(config)
    result = CliRunner().invoke(app, ["search", str(config)])
    assert result.exit_code == 2
    assert "--state is required" in result.output
