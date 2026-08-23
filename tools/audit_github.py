"""Audit the materialized GitHub roadmap against the canonical catalog."""

from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from tools.bootstrap_github import (
    ISSUES,
    LABELS,
    MILESTONES,
    OWNER,
    PROJECT_TITLE,
    REPOSITORY,
    TYPE_LABEL,
    gh_json,
    graphql,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "work" / "github_audit_report.json"
TOPOLOGY = ROOT / "work" / "dependency_topology.txt"
REQUIRED_SECTIONS = (
    "## Summary", "## Motivation", "## Scope", "## Non-goals",
    "## Proposed implementation", "## Acceptance criteria", "## Tests",
    "## Documentation impact", "## Dependencies", "## Notes / risks",
)


def all_issues() -> dict[str, dict[str, Any]]:
    values = gh_json("api", "--paginate", f"repos/{REPOSITORY}/issues?state=all&per_page=100")
    result: dict[str, dict[str, Any]] = {}
    for value in values:
        match = re.search(r"<!-- roadmap-key:([a-z0-9-]+) -->", value.get("body") or "")
        if match:
            result[match.group(1)] = value
    return result


def project() -> dict[str, Any]:
    values = graphql(
        "query($login:String!){user(login:$login){projectsV2(first:100){nodes{id number title url}}}}",
        {"login": OWNER},
    )["user"]["projectsV2"]["nodes"]
    return next(value for value in values if value["title"] == PROJECT_TITLE)


def project_items(project_id: str) -> dict[int, dict[str, str]]:
    query = """query($id:ID!,$cursor:String){node(id:$id){... on ProjectV2{
      items(first:100,after:$cursor){nodes{content{... on Issue{number}}
      fieldValues(first:30){nodes{... on ProjectV2ItemFieldSingleSelectValue{
      name field{... on ProjectV2FieldCommon{name}}}}}}pageInfo{hasNextPage endCursor}}}}}"""
    cursor: str | None = None
    result: dict[int, dict[str, str]] = {}
    while True:
        values = graphql(query, {"id": project_id, "cursor": cursor})["node"]["items"]
        for item in values["nodes"]:
            content = item.get("content")
            if not content or "number" not in content:
                continue
            fields: dict[str, str] = {}
            for value in item["fieldValues"]["nodes"]:
                if value and value.get("field"):
                    fields[value["field"]["name"]] = value["name"]
            result[int(content["number"])] = fields
        if not values["pageInfo"]["hasNextPage"]:
            return result
        cursor = values["pageInfo"]["endCursor"]


def native_dependencies(issue: dict[str, Any]) -> tuple[str, set[int]]:
    values = gh_json(
        "api", "-H", "X-GitHub-Api-Version: 2026-03-10",
        f"repos/{REPOSITORY}/issues/{issue['number']}/dependencies/blocked_by?per_page=100",
    )
    match = re.search(r"<!-- roadmap-key:([a-z0-9-]+) -->", issue["body"])
    assert match
    return match.group(1), {int(value["number"]) for value in values}


def audit() -> dict[str, Any]:
    errors: list[str] = []
    issues = all_issues()
    expected_keys = {spec.key for spec in ISSUES}
    if set(issues) != expected_keys:
        errors.append(f"issue key mismatch: missing={sorted(expected_keys-set(issues))} extra={sorted(set(issues)-expected_keys)}")
    numbers = {key: int(value["number"]) for key, value in issues.items()}

    for spec in ISSUES:
        value = issues[spec.key]
        body = value.get("body") or ""
        for section in REQUIRED_SECTIONS:
            if section not in body:
                errors.append(f"#{value['number']} missing section {section}")
        actual_labels = {label["name"] for label in value["labels"]}
        required_labels = {TYPE_LABEL[spec.kind], f"area:{spec.area}", f"priority:{spec.priority}", f"size:{spec.effort}"}
        if not required_labels <= actual_labels:
            errors.append(f"#{value['number']} missing labels {sorted(required_labels-actual_labels)}")
        if value.get("milestone", {}).get("title") != spec.milestone:
            errors.append(f"#{value['number']} milestone mismatch")
        for dependency in spec.dependencies:
            if f"- #{numbers[dependency]}" not in body:
                errors.append(f"#{value['number']} body missing blocked-by #{numbers[dependency]}")

    with ThreadPoolExecutor(max_workers=8) as executor:
        native = dict(executor.map(native_dependencies, issues.values()))
    for spec in ISSUES:
        expected = {numbers[key] for key in spec.dependencies}
        if native[spec.key] != expected:
            errors.append(f"#{numbers[spec.key]} native dependencies expected={sorted(expected)} actual={sorted(native[spec.key])}")

    roadmap = project()
    items = project_items(roadmap["id"])
    for spec in ISSUES:
        number = numbers[spec.key]
        if number not in items:
            errors.append(f"#{number} missing from Project")
            continue
        expected_fields = {
            "Status": "Blocked" if spec.dependencies else "Ready",
            "Phase": spec.phase,
            "Priority": spec.priority,
            "Work Type": spec.kind,
            "Effort": spec.effort,
            "Risk": spec.risk,
        }
        for name, expected in expected_fields.items():
            if items[number].get(name) != expected:
                errors.append(f"#{number} Project {name}: expected {expected}, got {items[number].get(name)}")

    milestones = gh_json("api", f"repos/{REPOSITORY}/milestones?state=all&per_page=100")
    labels = gh_json("api", "--paginate", f"repos/{REPOSITORY}/labels?per_page=100")
    if not {name for name, _ in MILESTONES} <= {value["title"] for value in milestones}:
        errors.append("milestone catalog is incomplete")
    if not set(LABELS) <= {value["name"] for value in labels}:
        errors.append("label catalog is incomplete")

    topology = [f"{index:03d} #{numbers[spec.key]} {spec.key}" for index, spec in enumerate(ISSUES, 1)]
    TOPOLOGY.write_text("\n".join(topology) + "\n", encoding="utf-8")
    result = {
        "ok": not errors,
        "errors": errors,
        "repository": f"https://github.com/{REPOSITORY}",
        "project": roadmap["url"],
        "issues": len(issues),
        "native_dependency_edges": sum(len(value) for value in native.values()),
        "project_items": len(items),
        "milestones": len({value["title"] for value in milestones} & {name for name, _ in MILESTONES}),
        "roadmap_labels": len({value["name"] for value in labels} & set(LABELS)),
        "ready": sum(1 for spec in ISSUES if not spec.dependencies),
        "blocked": sum(1 for spec in ISSUES if spec.dependencies),
        "topological_order": str(TOPOLOGY),
    }
    REPORT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = audit()
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["ok"] else 1)
