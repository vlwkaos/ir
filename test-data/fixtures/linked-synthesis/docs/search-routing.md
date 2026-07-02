---
related:
  - ../src/search.rs#route_query
  - docs/search-routing.md
aliases:
  - routing shortcut
---

# Search Routing

The routing-shortcut policy returns local BM25 evidence immediately when the
top lexical score is decisive. This preserves latency while the semantic daemon
warms in the background.

The synthesis answer must mention both facts: decisive BM25 evidence and daemon
warmup.
