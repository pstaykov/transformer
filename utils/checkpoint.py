import numpy as np


def count_params(d):
    """Total scalar parameter count in a nested params() dict."""
    total = 0
    for v in d.values():
        if isinstance(v, dict):
            total += count_params(v)
        else:
            total += int(np.size(v))
    return total


def flatten(d, prefix=""):
    """Flatten a nested dict of arrays/scalars into {"a.b.c": value} form."""
    flat = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(flatten(v, prefix=key + "."))
        else:
            flat[key] = v
    return flat


def unflatten(flat):
    """Inverse of flatten(): {"a.b.c": value} -> nested dict."""
    nested = {}
    for key, v in flat.items():
        parts = key.split(".")
        cur = nested
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v
    return nested


def save_checkpoint(path, model, step, extra=None):
    """Save model params plus training metadata to a .npz file."""
    flat = flatten(model.params())
    payload = {k: np.asarray(v) for k, v in flat.items()}
    payload["_step"] = np.asarray(step)
    if extra:
        for k, v in extra.items():
            payload[f"_meta_{k}"] = np.asarray(v)
    np.savez(path, **payload)


def load_checkpoint(path, model):
    """Load a .npz checkpoint into model in-place. Returns (step, meta_dict)."""
    data = np.load(path, allow_pickle=True)
    flat, meta = {}, {}
    for key in data.files:
        if key == "_step":
            continue
        if key.startswith("_meta_"):
            meta[key[len("_meta_"):]] = data[key].item() if data[key].ndim == 0 else data[key]
            continue
        flat[key] = data[key]
    model.load_params(unflatten(flat))
    step = int(data["_step"]) if "_step" in data.files else 0
    return step, meta
