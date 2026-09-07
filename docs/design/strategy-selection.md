# Deterministic strategy selection

The strategy selector is an auditable policy boundary for choosing baselines,
surgeon, acquisition, search policy, and evaluation budget. It deliberately
remains rule-based until held-out meta-evidence has been validated; an opaque
learned selector cannot silently replace a strong baseline.

Every request includes the objective-contract ID, candidate-space ID, tool
revision, seed, candidate count, evaluation and cost budgets, required strong
baseline names, and capability-bearing options. The decision ID hashes the
canonical request and selected result, so equivalent inputs produce equivalent
decisions.

Required strong baselines are checked first. Missing, incompatible, or weakly
marked baselines produce explicit unsupported or failed outcomes and never get
substituted. For surgeon, acquisition, and search-policy kinds, compatible
options are ranked by deterministic estimated cost, uncertainty, evidence
level, and name. The selector retains alternatives and reasons in the result.

An override must name an available compatible option and carry an explicit
approval flag. Required baselines and every selected strategy are included in
the cost envelope; exceeding it fails closed before execution.
