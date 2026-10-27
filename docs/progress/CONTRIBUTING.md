# Maintaining the progress inventory

`modelsurgeon-progress-v1.json` is the sole source for project-progress scores, research/implementation maturity, benchmark entries, and representative compatibility cells. Every milestone has its own research question, declared scope, acceptance criteria, evidence, and implementation/effectiveness/readiness state. The JSON Schema is in [`schema-v1.json`](schema-v1.json); the standard-library validator in [`tools/progress.py`](../../tools/progress.py) additionally checks repository evidence paths, unique IDs, workstream and milestone weights, history, outcomes, dependency cycles, PR/integration claims, and generated output.

## Update workflow

For every meaningful research, implementation, compatibility, or benchmark change:

1. Update the affected stable milestone(s) in the canonical JSON. Preserve prior IDs and append an assessment-history event rather than deleting a conclusion.
2. Keep research stage/outcome separate from implementation stage, effectiveness, and readiness. A negative conclusion can complete the explicitly bounded research question without indicating an effective feature. Do not close a broader research avenue because one experiment finished; split later questions into new milestones and retain old IDs/history.
3. Attach local evidence references with exact scope and limitations. Use an external-only source record when evidence cannot be retained locally; do not promote it as a completed experiment.
4. For upstream research, pin the exact revision, record inspected source paths and the actual applicable license, and append a research-source history event. An unknown or absent code license blocks incorporation; a model-weight license is not automatically a code license.
5. Add newly discovered avenues to the existing workstream or create a versioned scope change. Update weights transparently; scope growth can lower completion.
6. Record open PRs as pending. `integrated` and `integrated_main` require the assessed main revision, not a branch or a green PR. Refresh the dated backlog snapshot when taking a new assessment.
7. Run `uv run python tools/progress.py validate`, then `uv run python tools/progress.py generate`.
8. Commit the JSON and all generated Markdown with the code/evidence change. Run `uv run python tools/progress.py check` before submission.

CI runs validation and check-only generation. It reports stale output and never commits generated changes. Full CPU/GPU experiments may run outside CI; record their pinned model/dataset, inputs, source commit, config, seeds, hardware/software, baseline/candidate metrics, uncertainty, artifact hashes, quality gates, limitations, and reproduction command before advancing a physical-research stage.

## Scope, compatibility, and scoring

The assessment is versioned and deliberately grouped into non-Cartesian workstreams. `compatibility_cells` are separately weighted representative architecture/model type × format × quantization × operation × backend × runtime × recovery examples; unknown combinations remain visible and unclaimed. Each cell independently records theoretical plausibility, implementation location (`in_pr` is not integrated), and validation level (`synthetic` is not physical validation). `verified` requires a plausible scope, main integration, and physical or independent reproduction evidence. The scoring method, fixed dimension weights, status scale, and current overall assessment are generated in [`README.md`](README.md). A score is a transparent coverage estimate for this declared scope, not a certification or a competitiveness claim.

Research sources record the exact upstream revision and license only when verified. Unpinned/unverified sources are investigation backlog. Direct code incorporation requires a separate source-file and license audit, provenance, and notice preservation.
