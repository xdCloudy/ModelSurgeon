# Active-learning experiment budgets

Issue #113 gates every evaluation before its candidate boundary using estimated attempt
count, wall seconds, tier cost, GPU seconds, and disk bytes. If any dimension would exceed
its limit, no reservation is created and all exhausted dimensions are reported. Only one
reservation may be active, preventing concurrent work from silently oversubscribing a
sequential ledger.

Completion records observed resource use and success/failure counts. Failed-attempt policy
is explicit and versioned:

- `release-unused-charge-observed` charges the failed attempt count and observed resources,
  releasing the unused portion of its reservation.
- `charge-full-reservation` charges at least every reserved resource, even when failure
  occurs early.

The default is observed charging. Successful attempts always charge observed use. Snapshots
retain the policy, counts, accumulated resources, and any active candidate.
