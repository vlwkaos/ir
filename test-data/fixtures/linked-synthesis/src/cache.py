# [cache-key-model-version]
# [[Cache Policy]] explains why the model version cannot be omitted.
def cache_key(model_version: str, text_hash: str) -> str:
    return f"{model_version}:{text_hash}"
