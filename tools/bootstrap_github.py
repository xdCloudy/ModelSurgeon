"""Idempotently materialize the ModelSurgeon roadmap in GitHub.

Requires an authenticated `gh` CLI token with `repo`, `workflow`, and `project` scopes.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from tools.roadmap_catalog import ISSUES, MILESTONES, PHASES

OWNER = "xdCloudy"
REPO = "ModelSurgeon"
REPOSITORY = f"{OWNER}/{REPO}"
PROJECT_TITLE = "ModelSurgeon Roadmap"
ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "work" / "github_bootstrap_state.json"

LABELS: dict[str, tuple[str, str]] = {
    "type:feature": ("1f6feb", "User-visible or internal product capability"),
    "type:research": ("8250df", "Bounded research question with a decision outcome"),
    "type:bug": ("d73a4a", "Confirmed defect"),
    "type:testing": ("0e8a16", "Test coverage or quality validation"),
    "type:docs": ("0075ca", "Documentation or release communication"),
    "type:infra": ("5319e7", "Infrastructure, storage, tooling or automation"),
    "type:performance": ("fbca04", "Runtime, memory or I/O performance"),
    "priority:P0": ("b60205", "Critical path or architecture prerequisite"),
    "priority:P1": ("d93f0b", "Required for the next milestone deliverable"),
    "priority:P2": ("fbca04", "Important but not on the primary research path"),
    "priority:P3": ("cfd3d7", "Future or optional enhancement"),
    "size:XS": ("d4c5f9", "Up to roughly half a focused engineering day"),
    "size:S": ("bfdadc", "Roughly half to one focused engineering day"),
    "size:M": ("7fdbda", "Roughly one to one-and-a-half focused engineering days"),
    "size:L": ("c2e0c6", "Roughly one-and-a-half to two focused engineering days"),
    "hardware:cpu": ("e4e669", "CPU behavior or optimization"),
    "hardware:cuda": ("76d7c4", "CUDA behavior or optimization"),
    "hardware:memory": ("fef2c0", "RAM, VRAM, storage or out-of-core behavior"),
    "status:blocked": ("6e7781", "Cannot start until declared dependencies are complete"),
    "good-first-issue": ("7057ff", "Approachable, well-bounded first contribution"),
    "help-wanted": ("008672", "Maintainers welcome contributor ownership"),
}

for area in (
    "adapters", "graph", "features", "instrumentation", "surgery", "evaluation",
    "experiments", "dataset", "surgeon", "active-learning", "search",
    "explainability", "cli", "ci", "docs",
):
    LABELS[f"area:{area}"] = ("c5def5", f"Work owned by the {area} subsystem")

TYPE_LABEL = {
    "Feature": "type:feature",
    "Research": "type:research",
    "Bug": "type:bug",
    "Testing": "type:testing",
    "Documentation": "type:docs",
    "Infrastructure": "type:infra",
    "Performance": "type:performance",
}

FIELD_OPTIONS = {
    "Status": ["Backlog", "Ready", "In Progress", "Blocked", "In Review", "Done"],
    "Phase": list(PHASES),
    "Priority": ["P0", "P1", "P2", "P3"],
    # GitHub reserves the exact name "Type" for its built-in item type, even when that
    # field is unavailable on a user-owned project. "Work Type" is the closest editable field.
    "Work Type": ["Feature", "Research", "Infrastructure", "Testing", "Documentation", "Performance", "Bug"],
    "Effort": ["XS", "S", "M", "L"],
    "Risk": ["Low", "Medium", "High", "Research"],
}

FIELD_COLORS = ["GRAY", "BLUE", "GREEN", "YELLOW", "ORANGE", "RED", "PURPLE", "PINK"]


def run_gh(*args: str, input_text: str | None = None, check: bool = True) -> str:
    command = ["gh", *args]
    result = subprocess.run(
        command,
        input=input_text,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if check and result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n{result.stderr.strip()}"
        )
    return result.stdout


def gh_json(*args: str, input_value: Any | None = None) -> Any:
    text = run_gh(*args, input_text=None if input_value is None else json.dumps(input_value))
    return json.loads(text) if text.strip() else None


def graphql(query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"query": query}
    if variables:
        payload["variables"] = variables
    response = gh_json("api", "graphql", "--input", "-", input_value=payload)
    if response.get("errors"):
        raise RuntimeError(json.dumps(response["errors"], indent=2))
    return response["data"]


def ensure_repository_metadata() -> None:
    run_gh(
        "repo", "edit", REPOSITORY,
        "--description", "Learned automated structural optimization and pruning for neural networks",
        "--homepage", "https://github.com/xdCloudy/ModelSurgeon",
        "--enable-issues=true",
        "--enable-projects=true",
        "--enable-wiki=false",
    )
    run_gh(
        "repo", "edit", REPOSITORY,
        "--add-topic", "model-compression",
        "--add-topic", "neural-network-pruning",
        "--add-topic", "machine-learning",
        "--add-topic", "pytorch",
        "--add-topic", "transformers",
        "--add-topic", "gguf",
        "--add-topic", "llama-cpp",
        "--add-topic", "active-learning",
        "--add-topic", "consumer-hardware",
    )


def ensure_labels() -> None:
    for name, (color, description) in LABELS.items():
        run_gh(
            "label", "create", name, "--repo", REPOSITORY, "--color", color,
            "--description", description, "--force",
        )


def ensure_milestones() -> dict[str, int]:
    existing = gh_json("api", f"repos/{REPOSITORY}/milestones?state=all&per_page=100")
    by_title = {entry["title"]: entry for entry in existing}
    for title, description in MILESTONES:
        if title not in by_title:
            created = gh_json(
                "api", "--method", "POST", f"repos/{REPOSITORY}/milestones",
                "-f", f"title={title}", "-f", f"description={description}",
            )
            by_title[title] = created
        else:
            gh_json(
                "api", "--method", "PATCH",
                f"repos/{REPOSITORY}/milestones/{by_title[title]['number']}",
                "-f", f"description={description}",
            )
    return {title: int(by_title[title]["number"]) for title, _ in MILESTONES}


def render_body(spec: Any, numbers: dict[str, int], blocks: dict[str, list[str]]) -> str:
    blocked_lines = [f"- #{numbers[key]}" for key in spec.dependencies] or ["- None"]
    block_lines = [f"- #{numbers[key]}" for key in blocks[spec.key]] or ["- None"]
    acceptance = "\n".join(f"- [ ] {item}" for item in spec.acceptance)
    tests = "\n".join(f"- [ ] {item}" for item in spec.tests)
    dependency_scope = ", ".join(f"#{numbers[key]}" for key in spec.dependencies) or "none"
    return f"""<!-- roadmap-key:{spec.key} -->
## Summary

{spec.scope}

## Motivation

This bounded deliverable advances the **{spec.phase}** phase of **{spec.milestone}** and enables its declared downstream work without expanding into adjacent roadmap items.

## Scope

- {spec.scope}
- Produce the implementation artifact, validation evidence, and typed/versioned interfaces needed by direct dependants.

## Non-goals

- Work assigned to dependent or downstream issues is excluded.
- Broad redesign outside this subsystem is excluded unless a violated invariant is documented and approved.

## Proposed implementation

Implement behind the owning `area:{spec.area}` boundary, consume canonical component/config/provenance contracts, fail explicitly on unsupported inputs, and keep CPU and consumer-memory behavior observable. Use deterministic inputs and bounded allocations where the operation can scale with model size.

## Acceptance criteria

{acceptance}

## Tests

{tests}

## Documentation impact

- Update the relevant architecture, design, user, or research documentation when behavior or a public contract changes.
- Add a changelog entry if the issue changes a user-visible command, schema, format, or safety guarantee.

## Dependencies

Blocked by:
{chr(10).join(blocked_lines)}

Blocks:
{chr(10).join(block_lines)}

Dependency keys: `{', '.join(spec.dependencies) or 'none'}`. Resolved dependency issues: {dependency_scope}.

## Notes / risks

- Risk: **{spec.risk}**. Effort target: **{spec.effort}** (approximately 0.5–2 focused engineering days).
- Preserve source checkpoints and record model/dataset/tool revisions whenever applicable.
"""


def issue_labels(spec: Any) -> list[str]:
    values = [TYPE_LABEL[spec.kind], f"area:{spec.area}", f"priority:{spec.priority}", f"size:{spec.effort}"]
    if spec.dependencies:
        values.append("status:blocked")
    if spec.key in {"logging", "component-id-spec", "tiny-fixtures"}:
        values.append("good-first-issue")
    if spec.risk == "Research":
        values.append("help-wanted")
    if any(word in spec.title.lower() for word in ("cuda", "gpu", "vram", "mixed-precision")):
        values.append("hardware:cuda")
    if any(word in spec.title.lower() for word in ("memory", "ram", "disk", "stream", "gguf", "artifact")):
        values.append("hardware:memory")
    if "cpu" in spec.title.lower():
        values.append("hardware:cpu")
    return sorted(set(values))


def list_existing_issues() -> dict[str, dict[str, Any]]:
    issues = gh_json(
        "api", "--paginate", f"repos/{REPOSITORY}/issues?state=all&per_page=100"
    )
    by_key: dict[str, dict[str, Any]] = {}
    for issue in issues:
        if "pull_request" in issue:
            continue
        match = re.search(r"<!-- roadmap-key:([a-z0-9-]+) -->", issue.get("body") or "")
        if match:
            by_key[match.group(1)] = issue
    return by_key


def ensure_issues(milestones: dict[str, int]) -> dict[str, dict[str, Any]]:
    current = list_existing_issues()
    # First pass allocates all issue numbers so reciprocal body links can be rendered.
    for index, spec in enumerate(ISSUES, start=1):
        if spec.key in current:
            continue
        created = gh_json(
            "api", "--method", "POST", f"repos/{REPOSITORY}/issues", "--input", "-",
            input_value={
                "title": spec.title,
                "body": f"<!-- roadmap-key:{spec.key} -->\nRoadmap issue body is being linked.",
                "milestone": milestones[spec.milestone],
            },
        )
        current[spec.key] = created
        if index % 20 == 0:
            print(f"created/located {index}/{len(ISSUES)} issues", flush=True)

    numbers = {key: int(value["number"]) for key, value in current.items()}
    blocks: dict[str, list[str]] = defaultdict(list)
    for spec in ISSUES:
        for dependency in spec.dependencies:
            blocks[dependency].append(spec.key)

    # Second pass applies complete reciprocal dependency bodies, labels and milestones.
    for index, spec in enumerate(ISSUES, start=1):
        issue = current[spec.key]
        updated = gh_json(
            "api", "--method", "PATCH", f"repos/{REPOSITORY}/issues/{issue['number']}",
            "--input", "-",
            input_value={
                "title": spec.title,
                "body": render_body(spec, numbers, blocks),
                "milestone": milestones[spec.milestone],
                "state": "open",
                "labels": issue_labels(spec),
            },
        )
        current[spec.key] = updated
        if index % 20 == 0:
            print(f"updated {index}/{len(ISSUES)} issue bodies", flush=True)
    return current


def ensure_native_dependencies(issues: dict[str, dict[str, Any]]) -> int:
    edge_count = 0
    for spec in ISSUES:
        child_number = int(issues[spec.key]["number"])
        existing = gh_json(
            "api", "-H", "X-GitHub-Api-Version: 2026-03-10",
            f"repos/{REPOSITORY}/issues/{child_number}/dependencies/blocked_by?per_page=100",
        )
        existing_ids = {int(value["id"]) for value in existing}
        for dependency in spec.dependencies:
            blocker_id = int(issues[dependency]["id"])
            if blocker_id not in existing_ids:
                gh_json(
                    "api", "--method", "POST",
                    "-H", "X-GitHub-Api-Version: 2026-03-10",
                    f"repos/{REPOSITORY}/issues/{child_number}/dependencies/blocked_by",
                    "-F", f"issue_id={blocker_id}",
                )
                time.sleep(0.08)
            edge_count += 1
    return edge_count


def get_owner_and_repo_nodes() -> tuple[str, str]:
    data = graphql(
        "query($owner:String!,$repo:String!){user(login:$owner){id} repository(owner:$owner,name:$repo){id}}",
        {"owner": OWNER, "repo": REPO},
    )
    return str(data["user"]["id"]), str(data["repository"]["id"])


def ensure_project() -> dict[str, Any]:
    owner_id, repository_id = get_owner_and_repo_nodes()
    data = graphql(
        "query($login:String!){user(login:$login){projectsV2(first:100){nodes{id number title url}}}}",
        {"login": OWNER},
    )
    projects = data["user"]["projectsV2"]["nodes"]
    project = next((value for value in projects if value["title"] == PROJECT_TITLE), None)
    if project is None:
        project = graphql(
            "mutation($owner:ID!,$title:String!){createProjectV2(input:{ownerId:$owner,title:$title}){projectV2{id number title url}}}",
            {"owner": owner_id, "title": PROJECT_TITLE},
        )["createProjectV2"]["projectV2"]
    graphql(
        "mutation($id:ID!){updateProjectV2(input:{projectId:$id,shortDescription:\"Finite v1.0 roadmap for learned and native low-memory GGUF model surgery\",readme:\"Canonical dependency-ordered roadmap for ModelSurgeon. Ready items have no open roadmap dependencies; Blocked items expose native GitHub dependencies and reciprocal body links.\"}){projectV2{id}}}",
        {"id": project["id"]},
    )
    # The link mutation is idempotent at the relationship level.
    try:
        graphql(
            "mutation($project:ID!,$repo:ID!){linkProjectV2ToRepository(input:{projectId:$project,repositoryId:$repo}){repository{id}}}",
            {"project": project["id"], "repo": repository_id},
        )
    except RuntimeError as error:
        if "already" not in str(error).lower():
            raise
    return project


def project_fields(project_id: str) -> dict[str, Any]:
    data = graphql(
        "query($id:ID!){node(id:$id){... on ProjectV2{fields(first:50){nodes{... on ProjectV2FieldCommon{id name dataType} ... on ProjectV2SingleSelectField{options{id name}}}}}}}",
        {"id": project_id},
    )
    return {value["name"]: value for value in data["node"]["fields"]["nodes"]}


def set_single_select_options(field_id: str, name: str, options: list[str]) -> None:
    option_values = [
        {"name": value, "color": FIELD_COLORS[index % len(FIELD_COLORS)], "description": ""}
        for index, value in enumerate(options)
    ]
    graphql(
        "mutation($id:ID!,$name:String!,$options:[ProjectV2SingleSelectFieldOptionInput!]){updateProjectV2Field(input:{fieldId:$id,name:$name,singleSelectOptions:$options}){projectV2Field{... on ProjectV2SingleSelectField{id}}}}",
        {"id": field_id, "name": name, "options": option_values},
    )


def ensure_project_fields(project: dict[str, Any]) -> dict[str, Any]:
    fields = project_fields(project["id"])
    # Reconfigure GitHub's built-in Status options; create the remaining fields.
    set_single_select_options(fields["Status"]["id"], "Status", FIELD_OPTIONS["Status"])
    for name in ("Phase", "Priority", "Work Type", "Effort", "Risk"):
        fields = project_fields(project["id"])
        if name not in fields:
            options = ",".join(FIELD_OPTIONS[name])
            run_gh(
                "project", "field-create", str(project["number"]), "--owner", OWNER,
                "--name", name, "--data-type", "SINGLE_SELECT",
                "--single-select-options", options,
            )
        else:
            set_single_select_options(fields[name]["id"], name, FIELD_OPTIONS[name])
    return project_fields(project["id"])


def chunks(values: list[Any], size: int) -> list[list[Any]]:
    return [values[index:index + size] for index in range(0, len(values), size)]


def ensure_project_items(project: dict[str, Any], fields: dict[str, Any], issues: dict[str, dict[str, Any]]) -> dict[str, str]:
    data = graphql(
        "query($id:ID!,$cursor:String){node(id:$id){... on ProjectV2{items(first:100,after:$cursor){nodes{id content{... on Issue{id}}} pageInfo{hasNextPage endCursor}}}}}",
        {"id": project["id"], "cursor": None},
    )
    nodes = list(data["node"]["items"]["nodes"])
    page = data["node"]["items"]["pageInfo"]
    while page["hasNextPage"]:
        data = graphql(
            "query($id:ID!,$cursor:String){node(id:$id){... on ProjectV2{items(first:100,after:$cursor){nodes{id content{... on Issue{id}}} pageInfo{hasNextPage endCursor}}}}}",
            {"id": project["id"], "cursor": page["endCursor"]},
        )
        nodes.extend(data["node"]["items"]["nodes"])
        page = data["node"]["items"]["pageInfo"]
    item_by_content = {
        value["content"]["id"]: value["id"] for value in nodes if value.get("content")
    }

    missing = [spec for spec in ISSUES if issues[spec.key]["node_id"] not in item_by_content]
    for batch in chunks(missing, 20):
        aliases = []
        for index, spec in enumerate(batch):
            aliases.append(
                f'a{index}:addProjectV2ItemById(input:{{projectId:"{project["id"]}",contentId:"{issues[spec.key]["node_id"]}"}}){{item{{id}}}}'
            )
        result = graphql("mutation{" + " ".join(aliases) + "}")
        for index, spec in enumerate(batch):
            item_by_content[issues[spec.key]["node_id"]] = result[f"a{index}"]["item"]["id"]

    field_options = {
        name: {option["name"]: option["id"] for option in fields[name]["options"]}
        for name in FIELD_OPTIONS
    }
    for batch in chunks(ISSUES, 8):
        mutations: list[str] = []
        alias = 0
        for spec in batch:
            item_id = item_by_content[issues[spec.key]["node_id"]]
            values = {
                "Status": "Blocked" if spec.dependencies else "Ready",
                "Phase": spec.phase,
                "Priority": spec.priority,
                "Work Type": spec.kind,
                "Effort": spec.effort,
                "Risk": spec.risk,
            }
            for name, value in values.items():
                mutations.append(
                    f'u{alias}:updateProjectV2ItemFieldValue(input:{{projectId:"{project["id"]}",itemId:"{item_id}",fieldId:"{fields[name]["id"]}",value:{{singleSelectOptionId:"{field_options[name][value]}"}}}}){{projectV2Item{{id}}}}'
                )
                alias += 1
        graphql("mutation{" + " ".join(mutations) + "}")
    return {spec.key: item_by_content[issues[spec.key]["node_id"]] for spec in ISSUES}


def write_state(project: dict[str, Any], issues: dict[str, dict[str, Any]], edge_count: int) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "repository_url": f"https://github.com/{REPOSITORY}",
        "project_url": project["url"],
        "project_number": project["number"],
        "issue_count": len(ISSUES),
        "dependency_count": edge_count,
        "milestone_count": len(MILESTONES),
        "label_count": len(LABELS),
        "issues": {key: value["number"] for key, value in issues.items() if key in {s.key for s in ISSUES}},
    }
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    if "--project-only" in sys.argv:
        issues = list_existing_issues()
        missing = [spec.key for spec in ISSUES if spec.key not in issues]
        if missing:
            raise RuntimeError(f"project-only mode found missing roadmap issues: {missing}")
        project = ensure_project()
        fields = ensure_project_fields(project)
        ensure_project_items(project, fields, issues)
        edge_count = sum(len(spec.dependencies) for spec in ISSUES)
        write_state(project, issues, edge_count)
        print(json.dumps({
            "repository": f"https://github.com/{REPOSITORY}",
            "project": project["url"],
            "issues": len(ISSUES),
            "dependencies": edge_count,
            "labels": len(LABELS),
            "milestones": len(MILESTONES),
        }, indent=2))
        return
    ensure_repository_metadata()
    ensure_labels()
    milestones = ensure_milestones()
    issues = ensure_issues(milestones)
    edge_count = ensure_native_dependencies(issues)
    project = ensure_project()
    fields = ensure_project_fields(project)
    ensure_project_items(project, fields, issues)
    write_state(project, issues, edge_count)
    print(json.dumps({
        "repository": f"https://github.com/{REPOSITORY}",
        "project": project["url"],
        "issues": len(ISSUES),
        "dependencies": edge_count,
        "labels": len(LABELS),
        "milestones": len(MILESTONES),
    }, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"bootstrap failed: {error}", file=sys.stderr)
        raise
