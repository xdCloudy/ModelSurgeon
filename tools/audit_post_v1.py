"""Read back and audit the materialized ModelSurgeon v1.1-v2.0 roadmap."""

from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from tools.audit_github import project, project_items
from tools.bootstrap_github import REPOSITORY, gh_json
from tools.bootstrap_post_v1 import (
    REQUIRED_BODY_SECTIONS,
    blocks_by_key,
    issue_labels,
    list_roadmap_issues,
    validate_catalog,
)
from tools.post_v1_roadmap_catalog import ISSUES, MILESTONES
from tools.roadmap_catalog import ISSUES as LEGACY_ISSUES

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "work" / "post_v1_github_audit.json"
TOPOLOGY = ROOT / "work" / "post_v1_dependency_topology.txt"


def native_blockers(issue: dict[str, Any]) -> tuple[str, set[int]]:
    values = gh_json(
        "api", "-H", "X-GitHub-Api-Version: 2026-03-10",
        f"repos/{REPOSITORY}/issues/{issue['number']}/dependencies/blocked_by?per_page=100",
    )
    match = re.search(r"<!-- roadmap-key:([a-z0-9-]+) -->", issue.get("body") or "")
    if not match:
        raise RuntimeError(f"#{issue['number']} has no roadmap key")
    return match.group(1), {int(value["number"]) for value in values}


def cycle_nodes() -> list[str]:
    graph = {spec.key: set(spec.dependencies) for spec in [*LEGACY_ISSUES, *ISSUES]}
    visiting: set[str] = set()
    visited: set[str] = set()
    cycles: set[str] = set()

    def visit(key: str) -> None:
        if key in visiting:
            cycles.add(key)
            return
        if key in visited:
            return
        visiting.add(key)
        for dependency in graph.get(key, set()):
            visit(dependency)
        visiting.remove(key)
        visited.add(key)

    for key in graph:
        visit(key)
    return sorted(cycles)


def audit() -> dict[str, Any]:
    validate_catalog()
    errors: list[str] = []
    all_issues = list_roadmap_issues()
    post_keys = {spec.key for spec in ISSUES}
    legacy_keys = {spec.key for spec in LEGACY_ISSUES}
    missing = (post_keys | legacy_keys) - set(all_issues)
    if missing:
        errors.append(f"missing roadmap keys: {sorted(missing)}")
        raise RuntimeError(errors[0])
    numbers = {key: int(value["number"]) for key, value in all_issues.items()}
    blocks = blocks_by_key()
    open_by_key = {key: value["state"] == "open" for key, value in all_issues.items()}

    for spec in ISSUES:
        issue = all_issues[spec.key]
        body = issue.get("body") or ""
        for section in REQUIRED_BODY_SECTIONS:
            if section not in body:
                errors.append(f"#{issue['number']} missing {section}")
        expected_blocked = {numbers[key] for key in spec.dependencies}
        body_blocked = {int(value) for value in re.findall(r"(?m)^- #(\d+)$", body.split("Blocks:", 1)[0])}
        if body_blocked != expected_blocked:
            errors.append(f"#{issue['number']} body Blocked by mismatch")
        expected_blocks = {numbers[key] for key in blocks.get(spec.key, [])}
        blocks_section = body.split("Blocks:", 1)[1].split("\n\n", 1)[0]
        body_blocks = {int(value) for value in re.findall(r"(?m)^- #(\d+)$", blocks_section)}
        if body_blocks != expected_blocks:
            errors.append(f"#{issue['number']} body Blocks mismatch")
        blocked = any(open_by_key[key] for key in spec.dependencies)
        actual_labels = {value["name"] for value in issue["labels"]}
        expected_labels = set(issue_labels(spec, blocked))
        if actual_labels != expected_labels:
            errors.append(f"#{issue['number']} labels expected={sorted(expected_labels)} actual={sorted(actual_labels)}")
        if (issue.get("milestone") or {}).get("title") != spec.milestone:
            errors.append(f"#{issue['number']} milestone mismatch")

    for spec in LEGACY_ISSUES:
        expected_new = {numbers[key] for key in blocks.get(spec.key, [])}
        if not expected_new:
            continue
        body = all_issues[spec.key].get("body") or ""
        section = body.split("Blocks:", 1)[1].split("\n\nDependency keys:", 1)[0]
        actual = {int(value) for value in re.findall(r"(?m)^- #(\d+)$", section)}
        if not expected_new <= actual:
            errors.append(f"#{numbers[spec.key]} missing reciprocal legacy body links {sorted(expected_new-actual)}")

    with ThreadPoolExecutor(max_workers=8) as executor:
        native = dict(executor.map(native_blockers, (all_issues[spec.key] for spec in ISSUES)))
    for spec in ISSUES:
        expected = {numbers[key] for key in spec.dependencies}
        if native[spec.key] != expected:
            errors.append(f"#{numbers[spec.key]} native dependencies expected={sorted(expected)} actual={sorted(native[spec.key])}")

    roadmap = project()
    items = project_items(roadmap["id"])
    for spec in ISSUES:
        number = numbers[spec.key]
        expected = {
            "Status": "Blocked" if any(open_by_key[key] for key in spec.dependencies) else "Ready",
            "Phase": spec.phase, "Priority": spec.priority, "Work Type": spec.kind,
            "Effort": spec.effort, "Risk": spec.risk,
        }
        if number not in items:
            errors.append(f"#{number} missing from Project")
        else:
            for name, value in expected.items():
                if items[number].get(name) != value:
                    errors.append(f"#{number} Project {name}: expected {value}, got {items[number].get(name)}")

    cycles = cycle_nodes()
    if cycles:
        errors.append(f"dependency cycles: {cycles}")
    milestone_by_key = {spec.key: spec.milestone for spec in [*LEGACY_ISSUES, *ISSUES]}
    edges = [(dependency, spec.key) for spec in ISSUES for dependency in spec.dependencies]
    v1_edges = [edge for edge in edges if milestone_by_key[edge[0]].startswith("v1.0")]
    cross_edges = [edge for edge in edges if milestone_by_key[edge[0]] != milestone_by_key[edge[1]]]
    topology = [f"{index:03d} #{numbers[spec.key]} {spec.key}" for index, spec in enumerate(ISSUES, 1)]
    TOPOLOGY.parent.mkdir(parents=True, exist_ok=True)
    TOPOLOGY.write_text("\n".join(topology) + "\n", encoding="utf-8")
    result = {
        "ok": not errors, "errors": errors, "new_issues": len(ISSUES),
        "new_milestones": len(MILESTONES), "dependency_relationships": len(edges),
        "relationships_involving_existing_v1_0_issues": len(v1_edges),
        "cross_milestone_dependencies": len(cross_edges), "cycles": len(cycles),
        "project": roadmap["url"], "topological_order": str(TOPOLOGY),
    }
    REPORT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = audit()
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["ok"] else 1)
