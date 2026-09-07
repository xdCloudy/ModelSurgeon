# Offline evidence explorer v1

`modelsurgeon.explain.explorer` projects canonical evidence cells into a
self-contained static HTML artifact. It is a publication view, not an
experiment controller: every value is copied from an immutable cell record and
the generated page contains no network requests or mutable data queries.

Each cell retains its status (`supported`, `negative`, `unsupported`, `failed`,
or `unknown`), immutable source ID, metrics including explicit unknown values,
lineage, architecture diff, and limitation/failure reason. The Pareto summary
only considers measured supported/negative cells with finite cost and quality
metrics; excluded cells remain visible in the table.

The renderer uses an inline stylesheet and a small local filter script. User
controlled titles, IDs, reasons, metrics, and embedded JSON are escaped so
markup cannot become executable content. Source links are fragment links to
immutable IDs, never local filesystem paths. Generation is bounded by a
declared maximum cell count and uses sorted unique cell IDs for deterministic
output.

```text
modelsurgeon explorer evidence-cells.json --output explorer.html --json
```
