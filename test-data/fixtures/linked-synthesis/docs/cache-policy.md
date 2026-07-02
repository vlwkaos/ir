---
related:
  - ../src/cache.py#cache_key
aliases:
  - Cache Policy
---

# Cache Policy

The cache-key-model-version policy includes the model version in every cache
key. Without that version component, embeddings from incompatible dimensions can
be reused after a model switch.

The synthesis answer must mention both facts: model version in the key and
incompatible embedding dimensions.
