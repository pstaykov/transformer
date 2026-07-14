"""Read checkpoints from either trainer into the NumPy model's params dict.

The two trainers write different formats and, per the README, don't share
checkpoints: train.py writes .npz (see utils/checkpoint.py), while the CUDA
trainer writes its own little-endian binary (see cuda/include/checkpoint.cuh):

    magic "TFCKPT1\\n" (8 bytes)
    int32  step
    int32  vocab_size, d_model, num_heads, num_layers, d_ff, max_len
    int32  num_params
    repeat num_params:
        int32   name_len
        char[]  name
        int64   size          (element count)
        float[] data          (size float32s)

The .ckpt carries the architecture in its header; the .npz does not, and
num_heads in particular is not recoverable from the weight shapes alone (every
attention matrix is (d_model, d_model)), so loading a .npz needs num_heads
supplied by the caller.

Parameter names differ between the two in three places: the CUDA side wraps
every Linear as "<name>.W"/"<name>.b" (so attention is "attn.Wq.W" where numpy
has a bare "attn.Wq"), and it calls the MLP projections "hidden"/"output" where
numpy calls them "hidden_layer"/"output_layer".
"""

import os
import struct

import numpy as np

from utils.checkpoint import unflatten

MAGIC = b"TFCKPT1\n"


def _shape_for(name, cfg):
    """Expected shape of a CUDA-checkpoint param, which stores flat float arrays."""
    d_model, d_ff = cfg["d_model"], cfg["d_ff"]
    vocab_size, max_len = cfg["vocab_size"], cfg["max_len"]

    if name == "embedding.token_emb":
        return (vocab_size, d_model)
    if name == "embedding.pos_emb":
        return (max_len, d_model)
    if name == "W_out.W":
        return (d_model, vocab_size)
    if name.endswith(".gamma"):
        return (d_model,)
    if name.endswith(".swiglu.beta"):
        return ()
    if name.endswith((".attn.Wq.W", ".attn.Wk.W", ".attn.Wv.W", ".attn.Wo.W")):
        return (d_model, d_model)
    if name.endswith(".mlp.hidden.W"):
        return (d_model, 2 * d_ff)  # fused SwiGLU gate+value projection
    if name.endswith(".mlp.hidden.b"):
        return (2 * d_ff,)
    if name.endswith(".mlp.output.W"):
        return (d_ff, d_model)
    if name.endswith(".mlp.output.b"):
        return (d_model,)
    raise ValueError(f"unrecognized checkpoint param {name!r}")


def _to_numpy_key(name):
    """Rename a CUDA param to the key the NumPy model's load_params expects."""
    if name == "W_out.W":
        return "W_out"
    for cuda_name, np_name in ((".mlp.hidden.", ".mlp.hidden_layer."),
                               (".mlp.output.", ".mlp.output_layer.")):
        if cuda_name in name:
            return name.replace(cuda_name, np_name)
    # Attention projections are bias-free Linears on both sides; numpy stores
    # the matrix directly as "attn.Wq" rather than "attn.Wq.W".
    for proj in ("Wq", "Wk", "Wv", "Wo"):
        if name.endswith(f".attn.{proj}.W"):
            return name[: -len(".W")]
    return name


def load_cuda_ckpt(path):
    """Parse a CUDA .ckpt. Returns (params, config, step)."""
    with open(path, "rb") as f:
        blob = f.read()

    if blob[:8] != MAGIC:
        raise ValueError(f"{path} is not a CUDA checkpoint (bad magic)")

    off = 8
    step, vocab_size, d_model, num_heads, num_layers, d_ff, max_len, num_params = (
        struct.unpack_from("<8i", blob, off)
    )
    off += 32

    config = {
        "vocab_size": vocab_size,
        "d_model": d_model,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "d_ff": d_ff,
        "max_len": max_len,
    }

    flat = {}
    for _ in range(num_params):
        (name_len,) = struct.unpack_from("<i", blob, off)
        off += 4
        name = blob[off:off + name_len].decode("utf-8")
        off += name_len
        (size,) = struct.unpack_from("<q", blob, off)
        off += 8

        data = np.frombuffer(blob, dtype="<f4", count=size, offset=off).astype(np.float32)
        off += size * 4

        shape = _shape_for(name, config)
        expected = int(np.prod(shape)) if shape else 1
        if data.size != expected:
            raise ValueError(
                f"{path}: param {name} has {data.size} elements, expected {expected} {shape}"
            )
        # SwiGLU's beta rides along as a size-1 param but is a plain float.
        flat[_to_numpy_key(name)] = float(data[0]) if shape == () else data.reshape(shape)

    if off != len(blob):
        raise ValueError(f"{path}: {len(blob) - off} trailing bytes after {num_params} params")

    return unflatten(flat), config, step


def load_npz_ckpt(path, num_heads):
    """Parse a train.py .npz. Returns (params, config, step).

    The .npz stores no architecture metadata, so everything except num_heads is
    inferred from the weight shapes; num_heads cannot be, and must be passed in.
    """
    data = np.load(path, allow_pickle=True)
    flat = {k: data[k] for k in data.files if not k.startswith("_")}

    vocab_size, d_model = flat["embedding.token_emb"].shape
    max_len = flat["embedding.pos_emb"].shape[0]
    num_layers = 1 + max(
        int(k.split(".")[1]) for k in flat if k.startswith("blocks.")
    )
    d_ff = flat["blocks.0.mlp.hidden_layer.W"].shape[1] // 2

    config = {
        "vocab_size": int(vocab_size),
        "d_model": int(d_model),
        "num_heads": int(num_heads),
        "num_layers": int(num_layers),
        "d_ff": int(d_ff),
        "max_len": int(max_len),
    }
    step = int(data["_step"]) if "_step" in data.files else 0
    return unflatten(flat), config, step


def load_any(path, num_heads=8):
    """Load a .ckpt or .npz checkpoint. Returns (params, config, step).

    num_heads is only consulted for .npz files, where it isn't recoverable from
    the weights; a .ckpt carries the real value in its header.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".npz"):
        return load_npz_ckpt(path, num_heads)
    return load_cuda_ckpt(path)


def save_cuda_ckpt(path, params, config, step):
    """Write a CUDA-format .ckpt from a NumPy params dict.

    The CUDA trainer is the only thing that writes this format for real; this
    exists so the reader above can be round-trip tested without a GPU.
    """
    from utils.checkpoint import flatten

    flat = flatten(params)
    # Rebuild the CUDA traversal order from cuda/include/checkpoint.cuh.
    order = ["embedding.token_emb", "embedding.pos_emb"]
    for i in range(config["num_layers"]):
        p = f"blocks.{i}"
        order += [f"{p}.ln1.gamma", f"{p}.ln2.gamma"]
        order += [f"{p}.attn.{w}.W" for w in ("Wq", "Wk", "Wv", "Wo")]
        order += [f"{p}.mlp.hidden.W", f"{p}.mlp.hidden.b",
                  f"{p}.mlp.swiglu.beta",
                  f"{p}.mlp.output.W", f"{p}.mlp.output.b"]
    order += ["ln_f.gamma", "W_out.W"]

    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<8i", step, config["vocab_size"], config["d_model"],
                            config["num_heads"], config["num_layers"], config["d_ff"],
                            config["max_len"], len(order)))
        for cuda_name in order:
            value = np.asarray(flat[_to_numpy_key(cuda_name)], dtype=np.float32).ravel()
            raw = cuda_name.encode("utf-8")
            f.write(struct.pack("<i", len(raw)))
            f.write(raw)
            f.write(struct.pack("<q", value.size))
            f.write(value.tobytes())
