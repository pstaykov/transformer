#include "kernels.cuh"
#include "common.cuh"
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

namespace {

constexpr int BLOCK = 256;

// Block-wide reduction of `val` (sum) across all threads in the block.
// Every thread gets the reduced result back (broadcast via shared mem slot 0).
__device__ float block_reduce_sum(float val, float* shared) {
    int tid = threadIdx.x;
    shared[tid] = val;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        __syncthreads();
    }
    float result = shared[0];
    __syncthreads();
    return result;
}

__device__ float block_reduce_max(float val, float* shared) {
    int tid = threadIdx.x;
    shared[tid] = val;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        __syncthreads();
    }
    float result = shared[0];
    __syncthreads();
    return result;
}

__global__ void k_add_bias(float* y, const float* b, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    y[idx] += b[idx % cols];
}

__global__ void k_bias_grad(const float* dy, float* db, int rows, int cols) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cols) return;
    float sum = 0.0f;
    for (int r = 0; r < rows; r++) sum += dy[r * cols + c];
    db[c] = sum;
}

__global__ void k_sgd_update(float* param, const float* grad, float lr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    param[idx] -= lr * grad[idx];
}

__global__ void k_residual_add(float* out, const float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = a[idx] + b[idx];
}

__global__ void k_embedding_forward(const int* ids, const float* token_emb, const float* pos_emb,
                                     float* out, int B, int T, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T * D) return;
    int d = idx % D;
    int t = (idx / D) % T;
    int bt = idx / D;
    int tok = ids[bt];
    out[idx] = token_emb[tok * D + d] + pos_emb[t * D + d];
}

__global__ void k_embedding_backward(const int* ids, const float* dY, float* d_token_emb,
                                      float* d_pos_emb, int B, int T, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T * D) return;
    int d = idx % D;
    int t = (idx / D) % T;
    int bt = idx / D;
    int tok = ids[bt];
    atomicAdd(&d_token_emb[tok * D + d], dY[idx]);
    atomicAdd(&d_pos_emb[t * D + d], dY[idx]);
}

__global__ void k_rmsnorm_forward(const float* x, const float* gamma, float* out,
                                   float* rms_cache, int D, float eps) {
    extern __shared__ float shared[];
    int row = blockIdx.x;
    const float* xr = x + (size_t)row * D;
    float* outr = out + (size_t)row * D;

    float local = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) local += xr[d] * xr[d];
    float ss = block_reduce_sum(local, shared);

    float rms = sqrtf(ss / D + eps);
    if (threadIdx.x == 0) rms_cache[row] = rms;

    for (int d = threadIdx.x; d < D; d += blockDim.x) outr[d] = (xr[d] / rms) * gamma[d];
}

__global__ void k_rmsnorm_backward(const float* dOut, const float* x, const float* gamma,
                                    const float* rms_cache, float* dX, float* dGamma, int D) {
    extern __shared__ float shared[];
    int row = blockIdx.x;
    const float* xr = x + (size_t)row * D;
    const float* dOutr = dOut + (size_t)row * D;
    float* dXr = dX + (size_t)row * D;
    float rms = rms_cache[row];

    float local = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float xhat = xr[d] / rms;
        float dxhat = dOutr[d] * gamma[d];
        local += dxhat * xhat;
        atomicAdd(&dGamma[d], dOutr[d] * xhat);
    }
    float mean_term = block_reduce_sum(local, shared) / D;

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float xhat = xr[d] / rms;
        float dxhat = dOutr[d] * gamma[d];
        dXr[d] = (dxhat - xhat * mean_term) / rms;
    }
}

__global__ void k_swiglu_forward(const float* x, float beta, float* out, float* sig_cache, int Dff) {
    int row = blockIdx.x;
    const float* xr = x + (size_t)row * 2 * Dff;
    float* outr = out + (size_t)row * Dff;
    float* sigr = sig_cache + (size_t)row * Dff;
    for (int d = threadIdx.x; d < Dff; d += blockDim.x) {
        float gate = xr[d];
        float value = xr[Dff + d];
        float sig = 1.0f / (1.0f + expf(-beta * gate));
        sigr[d] = sig;
        outr[d] = gate * sig * value;
    }
}

__global__ void k_swiglu_backward(const float* dOut, const float* x, const float* sig_cache, float beta,
                                   float* dX, float* dBetaAccum, int Dff) {
    extern __shared__ float shared[];
    int row = blockIdx.x;
    const float* xr = x + (size_t)row * 2 * Dff;
    const float* dOutr = dOut + (size_t)row * Dff;
    const float* sigr = sig_cache + (size_t)row * Dff;
    float* dXr = dX + (size_t)row * 2 * Dff;

    float local_dbeta = 0.0f;
    for (int d = threadIdx.x; d < Dff; d += blockDim.x) {
        float gate = xr[d];
        float value = xr[Dff + d];
        float sig = sigr[d];
        float swish = gate * sig;
        float d_swish = sig + beta * swish * (1.0f - sig);

        dXr[d] = dOutr[d] * value * d_swish;              // d_gate
        dXr[Dff + d] = dOutr[d] * swish;                   // d_value
        local_dbeta += dOutr[d] * value * gate * gate * sig * (1.0f - sig);
    }
    float row_dbeta = block_reduce_sum(local_dbeta, shared);
    if (threadIdx.x == 0) atomicAdd(dBetaAccum, row_dbeta);
}

__global__ void k_softmax_forward(const float* x, float* out, int D) {
    extern __shared__ float shared[];
    int row = blockIdx.x;
    const float* xr = x + (size_t)row * D;
    float* outr = out + (size_t)row * D;

    float local_max = -FLT_MAX;
    for (int d = threadIdx.x; d < D; d += blockDim.x) local_max = fmaxf(local_max, xr[d]);
    float row_max = block_reduce_max(local_max, shared);

    float local_sum = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float e = expf(xr[d] - row_max);
        outr[d] = e;
        local_sum += e;
    }
    float row_sum = block_reduce_sum(local_sum, shared);

    for (int d = threadIdx.x; d < D; d += blockDim.x) outr[d] /= row_sum;
}

__global__ void k_softmax_backward(const float* dOut, const float* out, float* dX, int D) {
    extern __shared__ float shared[];
    int row = blockIdx.x;
    const float* dOutr = dOut + (size_t)row * D;
    const float* outr = out + (size_t)row * D;
    float* dXr = dX + (size_t)row * D;

    float local = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) local += dOutr[d] * outr[d];
    float dot = block_reduce_sum(local, shared);

    for (int d = threadIdx.x; d < D; d += blockDim.x) dXr[d] = outr[d] * (dOutr[d] - dot);
}

__global__ void k_causal_mask_add(float* scores, int batch, int T) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)batch * T * T;
    if (idx >= total) return;
    long long rem = idx % ((long long)T * T);
    int i = (int)(rem / T);
    int j = (int)(rem % T);
    if (j > i) scores[idx] += -1e9f;
}

__global__ void k_split_heads(const float* x, float* out, int B, int T, int H, int dh) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)B * H * T * dh;
    if (idx >= total) return;
    int d = idx % dh;
    int t = (idx / dh) % T;
    int h = (idx / dh / T) % H;
    int b = idx / dh / T / H;
    int D = H * dh;
    // x[b,t,h,d] -> out[b,h,t,d]
    out[idx] = x[((long long)b * T + t) * D + h * dh + d];
}

__global__ void k_merge_heads(const float* x, float* out, int B, int T, int H, int dh) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)B * H * T * dh;
    if (idx >= total) return;
    int d = idx % dh;
    int t = (idx / dh) % T;
    int h = (idx / dh / T) % H;
    int b = idx / dh / T / H;
    int D = H * dh;
    // x[b,h,t,d] -> out[b,t,h,d]
    out[((long long)b * T + t) * D + h * dh + d] = x[idx];
}

__global__ void k_cross_entropy_forward(const float* logits, const int* targets, float* probs_out,
                                         float* loss_sum, int* n_valid, int V, int ignore_index) {
    extern __shared__ float shared[];
    int row = blockIdx.x;
    const float* lr = logits + (size_t)row * V;
    float* pr = probs_out + (size_t)row * V;

    float local_max = -FLT_MAX;
    for (int v = threadIdx.x; v < V; v += blockDim.x) local_max = fmaxf(local_max, lr[v]);
    float row_max = block_reduce_max(local_max, shared);

    float local_sum = 0.0f;
    for (int v = threadIdx.x; v < V; v += blockDim.x) {
        float e = expf(lr[v] - row_max);
        pr[v] = e;
        local_sum += e;
    }
    float row_sum = block_reduce_sum(local_sum, shared);
    for (int v = threadIdx.x; v < V; v += blockDim.x) pr[v] /= row_sum;

    if (threadIdx.x == 0) {
        int target = targets[row];
        if (target != ignore_index) {
            float p = fmaxf(pr[target], 1e-12f);
            atomicAdd(loss_sum, -logf(p));
            atomicAdd(n_valid, 1);
        }
    }
}

// E4M3: 1 sign, 4 exponent, 3 mantissa bits. Constants mirror utils/fp8.py.
__constant__ float FP8_MAX_VAL = 448.0f;          // largest finite E4M3 magnitude
__constant__ float FP8_MIN_EXP = -6.0f;           // 1 - BIAS
__constant__ float FP8_MAX_EXP = 7.0f;            // (2^4 - 2) - BIAS
__constant__ float FP8_STEPS   = 8.0f;            // 2^MANTISSA_BITS

__global__ void k_quantize_e4m3(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = in[idx];
    if (v == 0.0f) { out[idx] = 0.0f; return; }
    float sign = v > 0.0f ? 1.0f : -1.0f;
    float av = fabsf(v);
    if (av > FP8_MAX_VAL) av = FP8_MAX_VAL;

    float e = floorf(log2f(av));
    if (e < FP8_MIN_EXP) e = FP8_MIN_EXP;
    else if (e > FP8_MAX_EXP) e = FP8_MAX_EXP;

    float scale = exp2f(e);
    float q = roundf(av / scale * FP8_STEPS) / FP8_STEPS * scale;
    out[idx] = sign * q;
}

__global__ void k_cross_entropy_backward(const float* probs, const int* targets, float* dLogits,
                                          int n_valid, int V, int ignore_index) {
    int row = blockIdx.x;
    const float* pr = probs + (size_t)row * V;
    float* dr = dLogits + (size_t)row * V;
    int target = targets[row];
    float inv_n = 1.0f / fmaxf((float)n_valid, 1.0f);

    for (int v = threadIdx.x; v < V; v += blockDim.x) {
        if (target == ignore_index) {
            dr[v] = 0.0f;
        } else {
            float grad = pr[v] - (v == target ? 1.0f : 0.0f);
            dr[v] = grad * inv_n;
        }
    }
}

} // namespace

void launch_add_bias(float* y, const float* b, int rows, int cols) {
    int n = rows * cols;
    k_add_bias<<<ceil_div(n, BLOCK), BLOCK>>>(y, b, rows, cols);
}

void launch_bias_grad(const float* dy, float* db, int rows, int cols) {
    k_bias_grad<<<ceil_div(cols, BLOCK), BLOCK>>>(dy, db, rows, cols);
}

void launch_sgd_update(float* param, const float* grad, float lr, int n) {
    k_sgd_update<<<ceil_div(n, BLOCK), BLOCK>>>(param, grad, lr, n);
}

void launch_residual_add(float* out, const float* a, const float* b, int n) {
    k_residual_add<<<ceil_div(n, BLOCK), BLOCK>>>(out, a, b, n);
}

void launch_embedding_forward(const int* ids, const float* token_emb, const float* pos_emb,
                               float* out, int B, int T, int D) {
    int n = B * T * D;
    k_embedding_forward<<<ceil_div(n, BLOCK), BLOCK>>>(ids, token_emb, pos_emb, out, B, T, D);
}

void launch_embedding_backward(const int* ids, const float* dY, float* d_token_emb,
                                float* d_pos_emb, int B, int T, int D) {
    int n = B * T * D;
    k_embedding_backward<<<ceil_div(n, BLOCK), BLOCK>>>(ids, dY, d_token_emb, d_pos_emb, B, T, D);
}

void launch_rmsnorm_forward(const float* x, const float* gamma, float* out,
                             float* rms_cache, int rows, int D, float eps) {
    int threads = min(D, BLOCK);
    k_rmsnorm_forward<<<rows, threads, threads * sizeof(float)>>>(x, gamma, out, rms_cache, D, eps);
}

void launch_rmsnorm_backward(const float* dOut, const float* x, const float* gamma,
                              const float* rms_cache, float* dX, float* dGamma, int rows, int D) {
    int threads = min(D, BLOCK);
    k_rmsnorm_backward<<<rows, threads, threads * sizeof(float)>>>(dOut, x, gamma, rms_cache, dX, dGamma, D);
}

void launch_swiglu_forward(const float* x, float beta, float* out, float* sig_cache, int rows, int Dff) {
    int threads = min(Dff, BLOCK);
    k_swiglu_forward<<<rows, threads>>>(x, beta, out, sig_cache, Dff);
}

void launch_swiglu_backward(const float* dOut, const float* x, const float* sig_cache, float beta,
                             float* dX, float* dBetaAccum, int rows, int Dff) {
    int threads = min(Dff, BLOCK);
    k_swiglu_backward<<<rows, threads, threads * sizeof(float)>>>(dOut, x, sig_cache, beta, dX, dBetaAccum, Dff);
}

void launch_softmax_forward(const float* x, float* out, int rows, int D) {
    int threads = min(D, BLOCK);
    k_softmax_forward<<<rows, threads, threads * sizeof(float)>>>(x, out, D);
}

void launch_softmax_backward(const float* dOut, const float* out, float* dX, int rows, int D) {
    int threads = min(D, BLOCK);
    k_softmax_backward<<<rows, threads, threads * sizeof(float)>>>(dOut, out, dX, D);
}

void launch_causal_mask_add(float* scores, int batch, int T) {
    long long n = (long long)batch * T * T;
    k_causal_mask_add<<<ceil_div((int)n, BLOCK), BLOCK>>>(scores, batch, T);
}

void launch_split_heads(const float* x, float* out, int B, int T, int H, int dh) {
    long long n = (long long)B * H * T * dh;
    k_split_heads<<<ceil_div((int)n, BLOCK), BLOCK>>>(x, out, B, T, H, dh);
}

void launch_merge_heads(const float* x, float* out, int B, int T, int H, int dh) {
    long long n = (long long)B * H * T * dh;
    k_merge_heads<<<ceil_div((int)n, BLOCK), BLOCK>>>(x, out, B, T, H, dh);
}

void launch_cross_entropy_forward(const float* logits, const int* targets, float* probs_out,
                                   float* loss_sum, int* n_valid, int rows, int V, int ignore_index) {
    int threads = min(V, BLOCK);
    k_cross_entropy_forward<<<rows, threads, threads * sizeof(float)>>>(
        logits, targets, probs_out, loss_sum, n_valid, V, ignore_index);
}

void launch_cross_entropy_backward(const float* probs, const int* targets, float* dLogits,
                                    int n_valid, int rows, int V, int ignore_index) {
    int threads = min(V, BLOCK);
    k_cross_entropy_backward<<<rows, threads>>>(probs, targets, dLogits, n_valid, V, ignore_index);
}

// ----------------------------------------------------------------------------
// FP8 simulation
// ----------------------------------------------------------------------------
bool g_fp8_enabled = false;

void launch_quantize_e4m3(const float* in, float* out, int n) {
    if (n <= 0) return;
    k_quantize_e4m3<<<ceil_div(n, BLOCK), BLOCK>>>(in, out, n);
}

namespace {
struct ScratchBuf {
    float* ptr = nullptr;
    size_t cap = 0;
    float* ensure(size_t n) {
        if (cap < n) {
            if (ptr) cudaFree(ptr);
            CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(float)));
            cap = n;
        }
        return ptr;
    }
    void release() {
        if (ptr) cudaFree(ptr);
        ptr = nullptr;
        cap = 0;
    }
};
ScratchBuf g_scratch_a, g_scratch_b;
}  // namespace

float* fp8_scratch_a(size_t n) { return g_scratch_a.ensure(n); }
float* fp8_scratch_b(size_t n) { return g_scratch_b.ensure(n); }

void fp8_free_scratch() {
    g_scratch_a.release();
    g_scratch_b.release();
}
