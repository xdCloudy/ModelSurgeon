"""Materialize the v1.1-v2.0 roadmap without mutating legacy issue state.

The catalog is validated before the first write. Existing v0.1-v1.0 issues are
only edited to add reciprocal ``Blocks`` links for new direct dependants.
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from tools.bootstrap_github import (
    FIELD_OPTIONS,
    OWNER,
    PROJECT_TITLE,
    REPOSITORY,
    TYPE_LABEL,
    chunks,
    gh_json,
    graphql,
    project_fields,
)
from tools.post_v1_roadmap_catalog import ISSUES, MILESTONES, PostV1IssueSpec
from tools.roadmap_catalog import ISSUES as LEGACY_ISSUES

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "work" / "post_v1_github_state.json"
REQUIRED_BODY_SECTIONS = (
    "## Summary",
    "## Motivation",
    "## Scope",
    "## Non-goals",
    "## Proposed implementation",
    "## Acceptance criteria",
    "## Benchmark / research protocol",
    "## Tests",
    "## Documentation impact",
    "## Dependencies",
    "## Risks",
    "## Completion evidence",
)


def catalog_by_key() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for spec in LEGACY_ISSUES:
        result[spec.key] = spec
    for spec in ISSUES:
        result[spec.key] = spec
    return result


def blocks_by_key() -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for spec in ISSUES:
        for dependency in spec.dependencies:
            result[dependency].append(spec.key)
    return result


def validate_catalog() -> dict[str, int]:
    errors: list[str] = []
    legacy_keys = {spec.key for spec in LEGACY_ISSUES}
    keys = [spec.key for spec in ISSUES]
    titles = [spec.title for spec in ISSUES]
    if len(keys) != len(set(keys)):
        errors.append("duplicate post-v1 roadmap keys")
    if len(titles) != len(set(titles)):
        errors.append("duplicate post-v1 issue titles")
    overlap = legacy_keys & set(keys)
    if overlap:
        errors.append(f"post-v1 keys overlap legacy keys: {sorted(overlap)}")

    milestone_order = {title: index for index, (title, _) in enumerate(MILESTONES)}
    counts: dict[str, int] = defaultdict(int)
    seen = set(legacy_keys)
    valid_phases = set(FIELD_OPTIONS["Phase"])
    for spec in ISSUES:
        counts[spec.milestone] += 1
        if spec.milestone not in milestone_order:
            errors.append(f"{spec.key}: unknown milestone {spec.milestone}")
        if spec.phase not in valid_phases:
            errors.append(f"{spec.key}: invalid Phase {spec.phase}")
        for field, value in (
            ("Priority", spec.priority), ("Work Type", spec.kind),
            ("Effort", spec.effort), ("Risk", spec.risk),
        ):
            if value not in FIELD_OPTIONS[field]:
                errors.append(f"{spec.key}: invalid {field} {value}")
        for dependency in spec.dependencies:
            if dependency not in seen:
                errors.append(f"{spec.key}: dangling or forward dependency {dependency}")
        if not spec.acceptance or not spec.protocol or not spec.completion:
            errors.append(f"{spec.key}: incomplete completion contract")
        seen.add(spec.key)
    for title, _ in MILESTONES:
        if not 6 <= counts[title] <= 15:
            errors.append(f"{title}: expected 6-15 issues, found {counts[title]}")
    if errors:
        raise RuntimeError("catalog validation failed:\n- " + "\n- ".join(errors))
    return dict(counts)


def list_roadmap_issues() -> dict[str, dict[str, Any]]:
    values = gh_json("api", "--paginate", f"repos/{REPOSITORY}/issues?state=all&per_page=100")
    result: dict[str, dict[str, Any]] = {}
    for value in values:
        if "pull_request" in value:
            continue
        match = re.search(r"<!-- roadmap-key:([a-z0-9-]+) -->", value.get("body") or "")
        if match:
            result[match.group(1)] = value
    return result


def issue_labels(spec: PostV1IssueSpec, blocked: bool) -> list[str]:
    labels = {
        TYPE_LABEL[spec.kind], f"area:{spec.area}",
        f"priority:{spec.priority}", f"size:{spec.effort}",
    }
    labels.update(f"hardware:{value}" for value in spec.hardware)
    if blocked:
        labels.add("status:blocked")
    if spec.risk == "Research":
        labels.add("help-wanted")
    return sorted(labels)


def render_body(
    spec: PostV1IssueSpec,
    numbers: dict[str, int],
    blocks: dict[str, list[str]],
) -> str:
    blocked_lines = [f"- #{numbers[key]}" for key in spec.dependencies] or ["- None"]
    block_lines = [f"- #{numbers[key]}" for key in blocks.get(spec.key, [])] or ["- None"]
    criteria = "\n".join(f"- [ ] {item}" for item in spec.acceptance)
    test_scope = (
        "unit tests for records, validation and deterministic decisions; integration tests for direct "
        "dependencies and public interfaces; regression tests for prior supported behavior; and "
        "real-artifact tests on licensed tiny fixtures or declared representative models when applicable"
    )
    research = (
        "For research-bearing work, preregister exact model families/checkpoints and revisions, "
        "datasets/tasks/splits and licenses, baselines/ablations, hardware and optimization/evaluation "
        "budgets, metrics and decision thresholds, random seeds, repeated-run confidence intervals, "
        "failure/unsupported-cell retention, and complete code/config/tool/artifact provenance. "
        "Do not turn a negative or inconclusive result into a success claim."
    )
    return f"""<!-- roadmap-key:{spec.key} -->
## Summary

{spec.deliverable}

## Motivation

This deliverable closes a concrete evidence or capability gap in **{spec.milestone}**. It is scoped so downstream issues can consume a versioned, testable result rather than an informal research assumption.

## Scope

- {spec.deliverable}
- Produce typed/versioned interfaces, retained evidence, and explicit supported, unsupported, failed, and unknown outcomes where relevant.
- Preserve source models and make cost, hardware, data, configuration, seed, and artifact lineage auditable.

## Non-goals

- {spec.non_goal}
- Work assigned to direct dependants is excluded from this issue.

## Proposed implementation

{spec.implementation}

Implementation must use the owning `area:{spec.area}` boundary, bounded resources, deterministic identifiers, explicit compatibility checks, and safe resume/rollback behavior where persistent artifacts are changed.

## Acceptance criteria

{criteria}

## Benchmark / research protocol

{spec.protocol}

{research}

## Tests

- [ ] Add {test_scope}.
- [ ] Exercise success, invalid-input, unsupported, interrupted/resumed, and negative-result paths relevant to this deliverable.
- [ ] Run the repository quality gate and record exact commands, environment, revisions, outcomes, and known skips.

## Documentation impact

- Update the applicable architecture/design document and public command/API/schema documentation.
- Add or update a research report with protocol, full results, negative cells, limitations, and reproduction commands when evidence is collected.
- Update `CHANGELOG.md`, compatibility tables, examples, and migration guidance for user-visible changes.

## Dependencies

Blocked by:
{chr(10).join(blocked_lines)}

Blocks:
{chr(10).join(block_lines)}

Dependency keys: `{', '.join(spec.dependencies) or 'none'}`.

## Risks

- Project risk: **{spec.risk}**. Effort: **{spec.effort}**. Priority: **{spec.priority}**.
- Primary risks include overfitting, benchmark leakage, unsupported-format assumptions, resource overruns, non-reproducible measurements, and misleading extrapolation; applicability depends on this issue's scope.
- Mitigate with held-out cells, hard budgets, compatibility gates, immutable evidence, confidence reporting, failure retention, and explicit claim boundaries.

## Completion evidence

- [ ] {spec.completion}
- [ ] Link implementation commits/PRs, test and benchmark logs, retained manifests/artifacts, documentation, and any negative or inconclusive results.
- [ ] Verify every acceptance item against the merged commit; prose, mock-only output, or an unreviewed local branch is not completion evidence.
"""


def ensure_milestones() -> dict[str, int]:
    existing = gh_json("api", "--paginate", f"repos/{REPOSITORY}/milestones?state=all&per_page=100")
    by_title = {value["title"]: value for value in existing}
    for title, description in MILESTONES:
        if title not in by_title:
            by_title[title] = gh_json(
                "api", "--method", "POST", f"repos/{REPOSITORY}/milestones",
                "-f", f"title={title}", "-f", f"description={description}",
            )
        elif by_title[title].get("description") != description:
            by_title[title] = gh_json(
                "api", "--method", "PATCH",
                f"repos/{REPOSITORY}/milestones/{by_title[title]['number']}",
                "-f", f"description={description}",
            )
    return {title: int(by_title[title]["number"]) for title, _ in MILESTONES}


def create_and_update_issues(
    milestones: dict[str, int], current: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    existing_titles = {value["title"]: key for key, value in current.items()}
    collisions = [(spec.key, existing_titles[spec.title]) for spec in ISSUES if spec.title in existing_titles and spec.key not in current]
    if collisions:
        raise RuntimeError(f"issue title collisions: {collisions}")
    for index, spec in enumerate(ISSUES, 1):
        if spec.key not in current:
            current[spec.key] = gh_json(
                "api", "--method", "POST", f"repos/{REPOSITORY}/issues", "--input", "-",
                input_value={"title": spec.title, "body": f"<!-- roadmap-key:{spec.key} -->\nLinking roadmap issue.", "milestone": milestones[spec.milestone]},
            )
        if index % 10 == 0:
            print(f"allocated {index}/{len(ISSUES)} post-v1 issues", flush=True)

    numbers = {key: int(value["number"]) for key, value in current.items()}
    blocks = blocks_by_key()
    open_by_key = {key: value["state"] == "open" for key, value in current.items()}
    for index, spec in enumerate(ISSUES, 1):
        blocked = any(open_by_key[key] for key in spec.dependencies)
        current[spec.key] = gh_json(
            "api", "--method", "PATCH", f"repos/{REPOSITORY}/issues/{numbers[spec.key]}", "--input", "-",
            input_value={
                "title": spec.title, "body": render_body(spec, numbers, blocks),
                "milestone": milestones[spec.milestone], "labels": issue_labels(spec, blocked),
            },
        )
        if index % 10 == 0:
            print(f"updated {index}/{len(ISSUES)} post-v1 issues", flush=True)
    return current


def update_legacy_blocks(current: dict[str, dict[str, Any]]) -> int:
    blocks = blocks_by_key()
    numbers = {key: int(value["number"]) for key, value in current.items()}
    changed = 0
    for spec in LEGACY_ISSUES:
        new_children = blocks.get(spec.key, [])
        if not new_children:
            continue
        issue = current[spec.key]
        body = issue.get("body") or ""
        match = re.search(r"(?ms)(^Blocks:\s*\n)(.*?)(?=\n\nDependency keys:)", body)
        if not match:
            raise RuntimeError(f"#{issue['number']} has no machine-editable Blocks section")
        existing = {int(value) for value in re.findall(r"^- #(\d+)$", match.group(2), re.MULTILINE)}
        merged = sorted(existing | {numbers[key] for key in new_children})
        replacement = match.group(1) + "\n".join(f"- #{number}" for number in merged)
        updated = body[:match.start()] + replacement + body[match.end():]
        if updated != body:
            gh_json(
                "api", "--method", "PATCH", f"repos/{REPOSITORY}/issues/{issue['number']}",
                "--input", "-", input_value={"body": updated},
            )
            changed += 1
    return changed


def sync_native_dependencies(current: dict[str, dict[str, Any]]) -> int:
    total = 0
    for index, spec in enumerate(ISSUES, 1):
        issue = current[spec.key]
        actual = gh_json(
            "api", "-H", "X-GitHub-Api-Version: 2026-03-10",
            f"repos/{REPOSITORY}/issues/{issue['number']}/dependencies/blocked_by?per_page=100",
        )
        expected_ids = {int(current[key]["id"]) for key in spec.dependencies}
        actual_ids = {int(value["id"]) for value in actual}
        for blocker_id in sorted(actual_ids - expected_ids):
            gh_json(
                "api", "--method", "DELETE", "-H", "X-GitHub-Api-Version: 2026-03-10",
                f"repos/{REPOSITORY}/issues/{issue['number']}/dependencies/blocked_by/{blocker_id}",
            )
        for blocker_id in sorted(expected_ids - actual_ids):
            gh_json(
                "api", "--method", "POST", "-H", "X-GitHub-Api-Version: 2026-03-10",
                f"repos/{REPOSITORY}/issues/{issue['number']}/dependencies/blocked_by",
                "-F", f"issue_id={blocker_id}",
            )
            time.sleep(0.05)
        total += len(expected_ids)
        if index % 10 == 0:
            print(f"linked {index}/{len(ISSUES)} post-v1 issues", flush=True)
    return total


def get_project() -> dict[str, Any]:
    projects = graphql(
        "query($login:String!){user(login:$login){projectsV2(first:100){nodes{id number title url}}}}",
        {"login": OWNER},
    )["user"]["projectsV2"]["nodes"]
    project = next((value for value in projects if value["title"] == PROJECT_TITLE), None)
    if project is None:
        raise RuntimeError(f"existing project {PROJECT_TITLE!r} not found")
    graphql(
        "mutation($id:ID!,$short:String!,$readme:String!){updateProjectV2(input:{projectId:$id,shortDescription:$short,readme:$readme}){projectV2{id}}}",
        {
            "id": project["id"],
            "short": "Dependency-ordered ModelSurgeon roadmap from foundations through autonomous v2.0 optimization",
            "readme": "Canonical roadmap for ModelSurgeon v0.1-v2.0. Status reflects open native GitHub blockers. Issues retain evidence, limitations, and reciprocal dependency links; the Project is the planning source of truth.",
        },
    )
    return dict(project)


def ensure_project_items(project: dict[str, Any], current: dict[str, dict[str, Any]]) -> None:
    fields = project_fields(project["id"])
    missing_fields = set(FIELD_OPTIONS) - set(fields)
    if missing_fields:
        raise RuntimeError(f"existing project is missing fields: {sorted(missing_fields)}")
    query = """query($id:ID!,$cursor:String){node(id:$id){... on ProjectV2{
      items(first:100,after:$cursor){nodes{id content{... on Issue{id}}}
      pageInfo{hasNextPage endCursor}}}}}"""
    cursor = None
    nodes: list[dict[str, Any]] = []
    while True:
        page = graphql(query, {"id": project["id"], "cursor": cursor})["node"]["items"]
        nodes.extend(page["nodes"])
        if not page["pageInfo"]["hasNextPage"]:
            break
        cursor = page["pageInfo"]["endCursor"]
    item_by_content = {value["content"]["id"]: value["id"] for value in nodes if value.get("content")}
    missing = [spec for spec in ISSUES if current[spec.key]["node_id"] not in item_by_content]
    for batch in chunks(missing, 20):
        add_mutations = " ".join(
            f'a{i}:addProjectV2ItemById(input:{{projectId:"{project["id"]}",contentId:"{current[spec.key]["node_id"]}"}}){{item{{id}}}}'
            for i, spec in enumerate(batch)
        )
        values = graphql("mutation{" + add_mutations + "}")
        for i, spec in enumerate(batch):
            item_by_content[current[spec.key]["node_id"]] = values[f"a{i}"]["item"]["id"]
    options = {name: {value["name"]: value["id"] for value in fields[name]["options"]} for name in FIELD_OPTIONS}
    open_by_key = {key: value["state"] == "open" for key, value in current.items()}
    for batch in chunks(ISSUES, 8):
        mutations: list[str] = []
        alias = 0
        for spec in batch:
            values = {
                "Status": "Blocked" if any(open_by_key[key] for key in spec.dependencies) else "Ready",
                "Phase": spec.phase, "Priority": spec.priority, "Work Type": spec.kind,
                "Effort": spec.effort, "Risk": spec.risk,
            }
            for name, value in values.items():
                mutations.append(
                    f'u{alias}:updateProjectV2ItemFieldValue(input:{{projectId:"{project["id"]}",itemId:"{item_by_content[current[spec.key]["node_id"]]}",fieldId:"{fields[name]["id"]}",value:{{singleSelectOptionId:"{options[name][value]}"}}}}){{projectV2Item{{id}}}}'
                )
                alias += 1
        graphql("mutation{" + " ".join(mutations) + "}")


def main() -> None:
    counts = validate_catalog()
    if "--validate-only" in sys.argv:
        print(json.dumps({"ok": True, "issues": len(ISSUES), "milestones": counts}, indent=2))
        return
    current = list_roadmap_issues()
    missing_legacy = {spec.key for spec in LEGACY_ISSUES} - set(current)
    if missing_legacy:
        raise RuntimeError(f"GitHub is missing legacy roadmap keys: {sorted(missing_legacy)}")
    milestones = ensure_milestones()
    current = create_and_update_issues(milestones, current)
    legacy_updates = update_legacy_blocks(current)
    edges = sync_native_dependencies(current)
    project = get_project()
    ensure_project_items(project, current)
    state = {
        "repository": f"https://github.com/{REPOSITORY}", "project": project["url"],
        "issues": len(ISSUES), "milestones": len(MILESTONES), "dependencies": edges,
        "legacy_bodies_updated": legacy_updates,
        "issue_numbers": {spec.key: current[spec.key]["number"] for spec in ISSUES},
    }
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(state | {"issue_numbers": "written to state file"}, indent=2))


if __name__ == "__main__":
    main()
