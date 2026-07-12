#pragma once
#include <cuda_runtime.h>

// All kernels operate on flat row-major float buffers. Launch helpers take
// plain dimensions (no shape metadata) - callers are the layer classes,
// which know the semantics.

void launch_add_bias(float* y, const float* b, int rows, int cols);
void launch_bias_grad(const float* dy, float* db, int rows, int cols);
void launch_sgd_update(float* param, const float* grad, float lr, int n);
void launch_residual_add(float* out, const float* a, const float* b, int n);

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
void launch_cross_entropy_forward(const float* logits, const int* targets, float* probs_out,
                                   float* loss_sum, int* n_valid, int rows, int V, int ignore_index);
void launch_cross_entropy_backward(const float* probs, const int* targets, float* dLogits,
                                    int n_valid, int rows, int V, int ignore_index);

// ---------------------------------------------------------------------------
// Simulated FP8 (E4M3), the CUDA counterpart of utils/fp8.py. Snaps each value
// onto the nearest representable E4M3 grid point (result kept in float32), so a
// following fp32 cuBLAS GEMM mirrors real FP8 tensor cores: FP8 operands, FP32
// accumulate. `g_fp8_enabled` is the global on/off switch (set from --fp8); the
// gemm() wrappers in gemm.cuh consult it before quantizing their operands into
// the reusable fp8_scratch_* buffers.
// ---------------------------------------------------------------------------
extern bool g_fp8_enabled;

void launch_quantize_e4m3(const float* in, float* out, int n);

// Reusable device scratch (grown on demand, never shrunk) for the two GEMM
// operands. Two separate buffers so a single GEMM can quantize A and B at once.
float* fp8_scratch_a(size_t n);
float* fp8_scratch_b(size_t n);
void fp8_free_scratch();
