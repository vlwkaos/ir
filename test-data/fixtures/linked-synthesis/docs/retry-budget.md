---
aliases:
  - Retry Budget
---

# Retry Budget

The retry-budget-jitter policy caps retries and adds jitter. This prevents
multiple workers from retrying at the same instant after a shared transient
failure.

The synthesis answer must mention both facts: capped retries and jitter avoids
synchronized retry spikes.
