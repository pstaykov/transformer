import numpy as np
from numba import njit, prange

"""Simulated FP8 (E4M3) numerics, JIT-compiled with numba.

No native 1-byte float dtype is available here (numpy has none built in, and
the `ml_dtypes` package that backs JAX/PyTorch FP8 support isn't installable
in this environment). Instead we simulate E4M3 by snapping float32 values
onto the nearest representable E4M3 grid point before every matmul, then
accumulate the product in float32 - this mirrors real FP8 tensor cores,
which take FP8 operands but accumulate in FP32.

The quantization step (elementwise log2/round over every matmul operand,
called for every weight matrix on every forward+backward pass) dominates
runtime for a 100M-parameter model in pure numpy. It's compiled to a
parallel machine-code loop with numba instead of numpy ufuncs, which avoids
numpy's intermediate-array allocations, multithreads across cores, and stays
in float32 the whole way through (benchmarked ~30% faster than upcasting to
float64 for the elementwise math, see utils/fp8_bench.py).
"""

MANTISSA_BITS = 3          # E4M3: 1 sign, 4 exponent, 3 mantissa bits
EXP_BITS = 4
BIAS = 7
MAX_VAL = np.float32(448.0)        # largest finite E4M3 magnitude
MIN_EXP = np.float32(1 - BIAS)     # smallest normal exponent
MAX_EXP = np.float32((2 ** EXP_BITS - 2) - BIAS)
STEPS = np.float32(2 ** MANTISSA_BITS)

ENABLED = True


@njit(parallel=True, fastmath=True, cache=True)
def _quantize_kernel(flat):
    out = np.empty_like(flat)
    n = flat.shape[0]
    for i in prange(n):
        v = flat[i]
        if v == 0.0:
            out[i] = np.float32(0.0)
            continue
        sign = np.float32(1.0) if v > 0.0 else np.float32(-1.0)
        av = v if v > 0.0 else -v
        if av > MAX_VAL:
            av = MAX_VAL

        e = np.floor(np.log2(av))
        if e < MIN_EXP:
            e = MIN_EXP
        elif e > MAX_EXP:
            e = MAX_EXP

        scale = np.float32(2.0) ** e
        q = np.round(av / scale * STEPS) / STEPS * scale
        out[i] = sign * q
    return out


def quantize(x):
    """Round a float array down to E4M3 precision (returned as float32)."""
    x32 = np.ascontiguousarray(x, dtype=np.float32)
    flat = _quantize_kernel(x32.ravel())
    return flat.reshape(x32.shape)


def matmul(a, b):
    """Matmul with both operands cast to FP8 precision, accumulated in float32."""
    if not ENABLED:
        return np.matmul(a, b)
    return np.matmul(quantize(a), quantize(b))


def qmatmul(a, bq):
    """Matmul where `a` still needs quantizing but `bq` is already FP8-quantized.

    Lets callers cache a quantized weight matrix across a forward/backward
    pair (weights don't change mid-step) instead of re-quantizing it on
    every matmul that touches it.
    """
    if not ENABLED:
        return np.matmul(a, bq)
    return np.matmul(quantize(a), bq)


def warmup():
    """Trigger numba JIT compilation ahead of time so the first real call isn't slow."""
    _quantize_kernel(np.array([1.0, -2.5, 0.0], dtype=np.float32))
