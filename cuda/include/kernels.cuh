#pragma once
#include <cuda_runtime.h>

// All kernels operate on flat row-major float buffers. Launch helpers take
// plain dimensions (no shape metadata) - callers are the layer classes,
// which know the semantics.

void launch_add_bias(float* y, const float* b, int rows, int cols);
void launch_bias_grad(const float* dy, float* db, int rows, int cols);
void launch_sgd_update(float* param, const float* grad, float lr, int n);
void launch_residual_add(float* out, const float* a, const float* b, int n);

// Adam/AdamW: m,v are per-parameter moment buffers (same size as param),
// bc1/bc2 are the (1 - beta^t) bias-correction denominators computed by the
// caller from the optimizer step count. weight_decay is decoupled (AdamW-style,
// applied directly to param rather than folded into the gradient).
void launch_adam_update(float* param, const float* grad, float* m, float* v,
                         float lr, float beta1, float beta2, float eps,
                         float bc1, float bc2, float weight_decay, int n);

// Sum of squares of `grad`, atomically accumulated into *accum (caller zeroes
// it first). Used to compute a global gradient norm across many buffers.
void launch_grad_sumsq(const float* grad, float* accum, int n);
// In-place elementwise scale, used to rescale gradients for global-norm clipping.
void launch_scale_inplace(float* buf, float scale, int n);

// Embedding: y[b,t,:] = token_emb[ids[b,t],:] + pos_emb[t,:]
void launch_embedding_forward(const int* ids, const float* token_emb, const float* pos_emb,
                               float* out, int B, int T, int D);
void launch_embedding_backward(const int* ids, const float* dY, float* d_token_emb,
                                float* d_pos_emb, int B, int T, int D);

// RMSNorm over the last dim (D) of `rows` independent rows.
void launch_rmsnorm_forward(const float* x, const float* gamma, float* out,
                             float* rms_cache, int rows, int D, float eps);
void launch_rmsnorm_backward(const float* dOut, const float* x, const float* gamma,
                              const float* rms_cache, float* dX, float* dGamma,
                              int rows, int D);

// SwiGLU: input (rows, 2*Dff) -> output (rows, Dff). Also accumulates dBeta.
void launch_swiglu_forward(const float* x, float beta, float* out, float* sig_cache, int rows, int Dff);
void launch_swiglu_backward(const float* dOut, const float* x, const float* sig_cache, float beta,
                             float* dX, float* dBetaAccum, int rows, int Dff);

// Softmax over the last dim (D) of `rows` independent rows.
void launch_softmax_forward(const float* x, float* out, int rows, int D);
void launch_softmax_backward(const float* dOut, const float* out, float* dX, int rows, int D);

// Additive causal mask: scores[..., i, j] += (j > i) ? -1e9 : 0, broadcast
// over a leading `batch` dimension (B*H), scores shaped (batch, T, T).
void launch_causal_mask_add(float* scores, int batch, int T);

// Split (B,T,D) -> (B,H,T,dh) / merge the inverse, D = H*dh.
void launch_split_heads(const float* x, float* out, int B, int T, int H, int dh);
void launch_merge_heads(const float* x, float* out, int B, int T, int H, int dh);

// Cross-entropy over the last dim (V) of `rows` rows. targets[row] == ignore_index skips that row.
// Returns (via out_loss_sum, out_n_valid device scalars) so the host can divide.
// label_smoothing in [0, 1): 0 reproduces plain cross-entropy. See k_cross_entropy_forward
// for the smoothed-target math.
void launch_cross_entropy_forward(const float* logits, const int* targets, float* probs_out,
                                   float* loss_sum, int* n_valid, int rows, int V, int ignore_index,
                                   float label_smoothing);
void launch_cross_entropy_backward(const float* probs, const int* targets, float* dLogits,
                                    int n_valid, int rows, int V, int ignore_index,
                                    float label_smoothing);

// ---------------------------------------------------------------------------
// Simulated FP8 (E4M3), the CUDA counterpart of utils/fp8.py. Snaps each value
// onto the nearest representable E4M3 grid point (result kept in float32), so a
// following fp32 cuBLAS GEMM mirrors real FP8 tensor cores: FP8 operands, FP32
// accumulate. `g_fp8_enabled` is the global on/off switch (set from --fp8); the
// gemm() wrappers in gemm.cuh consult it before quantizing their operands into
// the reusable fp8_scratch_* buffers.
// ---------------------------------------------------------------------------
extern bool g_fp8_enabled;

// ---------------------------------------------------------------------------
// Dropout. `g_training` gates it globally (off during eval/inference, e.g.
// probe.cu) the same way g_fp8_enabled gates the FP8 path above.
// ---------------------------------------------------------------------------
extern bool g_training;

// mask[idx] is 1/keep_prob or 0 (inverted dropout), cached for backward.
// `seed`/`offset` seed a per-element Philox stream so each call (and each
// element) gets independent randomness without a host round-trip.
void launch_dropout_forward(const float* x, float* y, float* mask, float keep_prob,
                             unsigned long long seed, unsigned long long offset, int n);
void launch_dropout_backward(const float* dOut, const float* mask, float* dX, int n);

void launch_quantize_e4m3(const float* in, float* out, int n);

// Reusable device scratch (grown on demand, never shrunk) for the two GEMM
// operands. Two separate buffers so a single GEMM can quantize A and B at once.
float* fp8_scratch_a(size_t n);
float* fp8_scratch_b(size_t n);
void fp8_free_scratch();
