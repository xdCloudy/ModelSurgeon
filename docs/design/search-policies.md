# Seeded search policies v1

Greedy, beam, and uncertainty-aware policies consume the same predicted objective
observations and hard-constraint observations. Predicted constraint failures are
never selected. Greedy chooses one highest predicted reward, beam chooses the
configured top width, and uncertainty-aware beam ranks reward plus a configurable
multiple of predicted reward uncertainty. The cumulative evaluation budget caps all
policies.

Tie ranks hash the seed, decision index, and candidate ID, so selection is stable
without hidden process RNG state. The serializable policy state records decision
index and every candidate already charged to the evaluation budget; resuming with
the same evidence therefore produces the same next decisions.

Every considered candidate receives a stored reason plus the full normalized
objective score/contributions, hard-constraint evaluation, predicted reward,
uncertainty, and acquisition score. Already-selected, unsafe, and cutoff candidates
remain visible rather than disappearing from the audit trail.
