#pragma once
#include <random>
#include <memory>
#include <vector>
#include "common.cuh"
#include "tensor.cuh"
#include "gemm.cuh"
#include "kernels.cuh"

// Host-side layer classes wired on top of the raw kernels/GEMM primitives in
// kernels.cuh / gemm.cuh. Mirrors the numpy reference implementation under
// utils/*.py 1:1 (same forward/backward math, same init scales). The FP8
// (fp8.py) path is available too: GEMMs that use fp8.matmul/qmatmul in the
// reference pass fp8=true to the gemm() wrappers, which snap their operands to
// E4M3 when the global g_fp8_enabled switch is on (--fp8). It's off by default,
// so without --fp8 everything runs in plain fp32.
//
// Every layer owns its own forward-output and gradient buffers as
// DeviceArray members and resizes them lazily via DeviceArray::ensure(), so
// repeated forward()/backward() calls across training steps (same B*T each
// time) don't reallocate. Layers cache whatever pointers/shapes they need
// from forward() to compute backward() without the caller re-supplying them,
// same as `self.x = x` in the Python classes.

// ---------------------------------------------------------------------------
// Linear: y = x @ W + b (b optional). W is (in_dim, out_dim) row-major.
// ---------------------------------------------------------------------------
struct Linear {
    int in_dim, out_dim;
    bool has_bias;
    float weight_decay;

    DeviceArray W, dW, mW, vW;
    DeviceArray b, db, mb, vb;

    DeviceArray out;  // (rows, out_dim)
    DeviceArray dX;   // (rows, in_dim)

    const float* x_cache = nullptr;
    int rows_cache = 0;

    // weight_decay is AdamW-style decoupled decay applied to W only (never to
    // the bias). Defaults to 0.01, the standard AdamW value for weight matrices.
    Linear(int in_dim, int out_dim, bool has_bias, float init_scale, std::mt19937& rng,
           float weight_decay = 0.01f);

    // Returns pointer to internal `out` buffer, shape (rows, out_dim).
    float* forward(cublasHandle_t handle, const float* x, int rows);
    // Returns pointer to internal `dX` buffer, shape (rows, in_dim).
    float* backward(cublasHandle_t handle, const float* dOut);

    // t is the Adam step count (1-indexed).
    void update(float lr, int t);

    void accum_grad_sumsq(DeviceArray& accum) const;
    void scale_grads(float s);
};

// ---------------------------------------------------------------------------
// RMSNorm over the last dim.
// ---------------------------------------------------------------------------
struct RMSNorm {
    int d_model;
    float eps;

    DeviceArray gamma, dGamma, mGamma, vGamma;
    DeviceArray out, rms_cache, dX;

    const float* x_cache = nullptr;
    int rows_cache = 0;

    explicit RMSNorm(int d_model, float eps = 1e-5f);

    float* forward(const float* x, int rows);
    float* backward(const float* dOut);

    void update(float lr, int t);

    void accum_grad_sumsq(DeviceArray& accum) const;
    void scale_grads(float s);
};

// ---------------------------------------------------------------------------
// SwiGLU: input (rows, 2*Dff) -> output (rows, Dff). `beta` is a learned scalar.
// ---------------------------------------------------------------------------
struct SwiGLU {
    int d_ff;
    float beta;
    float m_beta = 0.0f, v_beta = 0.0f;  // Adam state for the scalar beta (kept host-side)

    DeviceArray sig_cache, out, dX;
    DeviceArray dBetaAccum;  // single-element device scalar

    const float* x_cache = nullptr;
    int rows_cache = 0;

    explicit SwiGLU(int d_ff, float beta = 1.0f);

    float* forward(const float* x, int rows);
    float* backward(const float* dOut);

    void update(float lr, int t);

    void accum_grad_sumsq(DeviceArray& accum) const;
    void scale_grads(float s);
};

// ---------------------------------------------------------------------------
// MLP: Linear(d_model -> 2*d_ff) -> SwiGLU -> Linear(d_ff -> d_model)
// ---------------------------------------------------------------------------
struct MLP {
    Linear hidden_layer;
    SwiGLU swiglu;
    Linear output_layer;

    MLP(int d_model, int d_ff, std::mt19937& rng);

    float* forward(cublasHandle_t handle, const float* x, int rows);
    float* backward(cublasHandle_t handle, const float* dOut);

    void update(float lr, int t);

    void accum_grad_sumsq(DeviceArray& accum) const;
    void scale_grads(float s);
};

// ---------------------------------------------------------------------------
// Multi-head self-attention with a causal mask baked into forward().
// ---------------------------------------------------------------------------
struct MultiHeadAttention {
    int d_model, num_heads, d_head;

    Linear Wq, Wk, Wv, Wo;  // all d_model -> d_model, no bias

    // Per-head buffers, shape (B, H, T, d_head) flattened.
    DeviceArray Qh, Kh, Vh;
    DeviceArray scores;      // (B*H, T, T) raw Q@K^T/sqrt(dh) + causal mask
    DeviceArray probs;       // (B*H, T, T) softmax(scores), cached for backward
    DeviceArray attn_out;    // (B, H, T, d_head)
    DeviceArray merged;      // (B, T, D)

    DeviceArray dQh, dKh, dVh, dScores, dAttnOut;
    DeviceArray dMerged;  // reused as scratch for each of the Q/K/V merge-heads results
    DeviceArray dx;       // sum of the Q/K/V backward contributions, (B, T, D)

    int B_cache = 0, T_cache = 0;

    MultiHeadAttention(int d_model, int num_heads, std::mt19937& rng);

    // x: (B, T, D). Returns pointer to Wo's output buffer, (B, T, D).
    float* forward(cublasHandle_t handle, const float* x, int B, int T);
    // dOut: (B, T, D). Returns dX, (B, T, D).
    float* backward(cublasHandle_t handle, const float* dOut);

    void update(float lr, int t);

    void accum_grad_sumsq(DeviceArray& accum) const;
    void scale_grads(float s);
};

// ---------------------------------------------------------------------------
// Inverted dropout on a flat buffer. forward()/backward() are no-ops (return
// the input pointer unchanged) whenever !g_training or p<=0, so eval/inference
// callers (e.g. probe.cu) pay nothing extra as long as they leave g_training
// at its default or set it false themselves.
// ---------------------------------------------------------------------------
struct Dropout {
    float p;
    unsigned long long seed;
    unsigned long long call_counter = 0;
    bool active_cache = false;

    DeviceArray mask, out, dX;

    Dropout(float p, unsigned long long seed) : p(p), seed(seed) {}

    float* forward(const float* x, int n) {
        active_cache = g_training && p > 0.0f;
        if (!active_cache) return const_cast<float*>(x);
        out.ensure((size_t)n);
        mask.ensure((size_t)n);
        launch_dropout_forward(x, out.data, mask.data, 1.0f - p, seed, call_counter, n);
        ++call_counter;
        return out.data;
    }

    float* backward(const float* dOut, int n) {
        if (!active_cache) return const_cast<float*>(dOut);
        dX.ensure((size_t)n);
        launch_dropout_backward(dOut, mask.data, dX.data, n);
        return dX.data;
    }
};

// ---------------------------------------------------------------------------
// Pre-norm transformer block: x = x + drop(attn(ln1(x))); x = x + drop(mlp(ln2(x)))
// ---------------------------------------------------------------------------
struct TransformerBlock {
    RMSNorm ln1, ln2;
    MultiHeadAttention attn;
    MLP mlp;
    Dropout drop_attn, drop_mlp;

    DeviceArray x1;    // x + attn_out
    DeviceArray x2;    // x1 + mlp_out
    DeviceArray dx1;   // dOut + d(mlp branch)
    DeviceArray dx0;   // dx1 + d(attn branch)

    int B_cache = 0, T_cache = 0;

    TransformerBlock(int d_model, int num_heads, int d_ff, std::mt19937& rng, float dropout = 0.1f);

    float* forward(cublasHandle_t handle, const float* x, int B, int T);
    float* backward(cublasHandle_t handle, const float* dOut);

    void update(float lr, int t);

    void accum_grad_sumsq(DeviceArray& accum) const;
    void scale_grads(float s);
};

// ---------------------------------------------------------------------------
// Token + learned positional embedding. Leaf layer (backward has no dX).
// ---------------------------------------------------------------------------
struct Embedding {
    int vocab_size, d_model, max_len;

    DeviceArray token_emb, d_token_emb, m_token_emb, v_token_emb;
    DeviceArray pos_emb, d_pos_emb, m_pos_emb, v_pos_emb;
    DeviceArray out;  // (B, T, D)

    const int* ids_cache = nullptr;
    int B_cache = 0, T_cache = 0;

    Embedding(int vocab_size, int d_model, int max_len, std::mt19937& rng);

    float* forward(const int* ids, int B, int T);
    void backward(const float* dOut);  // fills d_token_emb / d_pos_emb

    // No weight decay on embeddings (standard practice; especially not on pos_emb).
    void update(float lr, int t);

    void accum_grad_sumsq(DeviceArray& accum) const;
    void scale_grads(float s);
};

// ---------------------------------------------------------------------------
// Full decoder-only transformer.
// ---------------------------------------------------------------------------
struct TransformerModel {
    int vocab_size, d_model, num_heads, num_layers, d_ff, max_len;

    Embedding embedding;
    Dropout drop_embed;
    std::vector<std::unique_ptr<TransformerBlock>> blocks;
    RMSNorm ln_f;
    Linear W_out;  // d_model -> vocab_size, no bias

    TransformerModel(int vocab_size, int d_model, int num_heads, int num_layers,
                      int d_ff, int max_len, std::mt19937& rng, float dropout = 0.1f);

    // ids: device int buffer, shape (B, T). Returns logits, (B, T, vocab_size).
    float* forward(cublasHandle_t handle, const int* ids, int B, int T);
    void backward(cublasHandle_t handle, const float* dLogits);

    void update(float lr, int t);

    // Global gradient-norm clipping across every parameter's gradient buffer,
    // computed right after backward() and before update(). Caps the size of any
    // single step so a batch of unusually spiky gradients can't blow up the
    // weights (this matters more now that update() also applies a fixed lr
    // schedule rather than the human eyeballing metrics.csv for a runaway loss).
    float grad_global_norm();
    void clip_grad_norm(float max_norm);
};

// ---------------------------------------------------------------------------
// Softmax cross-entropy over the vocab dim, with an ignorable target index.
// ---------------------------------------------------------------------------
struct CrossEntropyLoss {
    int ignore_index;
    float label_smoothing;

    DeviceArray probs;     // (rows, V)
    DeviceArray dLogits;   // (rows, V)
    DeviceArray loss_sum_d;
    DeviceIntArray n_valid_d;

    const int* targets_cache = nullptr;
    int rows_cache = 0, V_cache = 0, n_valid_cache = 0;

    explicit CrossEntropyLoss(int ignore_index = -100, float label_smoothing = 0.0f);

    // logits: (rows, V) device buffer, targets: (rows,) device int buffer.
    // Returns the scalar mean loss (copied back to host).
    float forward(const float* logits, const int* targets, int rows, int V);
    // Returns pointer to internal dLogits buffer, (rows, V).
    float* backward();
};
