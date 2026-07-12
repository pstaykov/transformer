#include "layers.cuh"
#include <algorithm>
#include <cmath>

// ============================================================================
// Linear
// ============================================================================
Linear::Linear(int in_dim, int out_dim, bool has_bias, float init_scale, std::mt19937& rng)
    : in_dim(in_dim), out_dim(out_dim), has_bias(has_bias) {
    W.randn_init((size_t)in_dim * out_dim, init_scale, rng);
    dW.alloc((size_t)in_dim * out_dim);
    if (has_bias) {
        b.fill_init(out_dim, 0.0f);
        db.alloc(out_dim);
    }
}

float* Linear::forward(cublasHandle_t handle, const float* x, int rows) {
    out.ensure((size_t)rows * out_dim);
    gemm(handle, Trans::N, Trans::N, rows, out_dim, in_dim, 1.0f, x, W.data, 0.0f, out.data, /*fp8=*/true);
    if (has_bias) launch_add_bias(out.data, b.data, rows, out_dim);
    x_cache = x;
    rows_cache = rows;
    return out.data;
}

float* Linear::backward(cublasHandle_t handle, const float* dOut) {
    // dW = x^T @ dOut, shape (in_dim, out_dim)
    gemm(handle, Trans::T, Trans::N, in_dim, out_dim, rows_cache, 1.0f, x_cache, dOut, 0.0f, dW.data);
    if (has_bias) launch_bias_grad(dOut, db.data, rows_cache, out_dim);

    // dX = dOut @ W^T, shape (rows, in_dim)
    dX.ensure((size_t)rows_cache * in_dim);
    gemm(handle, Trans::N, Trans::T, rows_cache, in_dim, out_dim, 1.0f, dOut, W.data, 0.0f, dX.data, /*fp8=*/true);
    return dX.data;
}

void Linear::update(float lr) {
    launch_sgd_update(W.data, dW.data, lr, (int)W.size);
    if (has_bias) launch_sgd_update(b.data, db.data, lr, (int)b.size);
}

// ============================================================================
// RMSNorm
// ============================================================================
RMSNorm::RMSNorm(int d_model, float eps) : d_model(d_model), eps(eps) {
    gamma.fill_init(d_model, 1.0f);
    dGamma.alloc(d_model);
}

float* RMSNorm::forward(const float* x, int rows) {
    out.ensure((size_t)rows * d_model);
    rms_cache.ensure(rows);
    launch_rmsnorm_forward(x, gamma.data, out.data, rms_cache.data, rows, d_model, eps);
    x_cache = x;
    rows_cache = rows;
    return out.data;
}

float* RMSNorm::backward(const float* dOut) {
    dX.ensure((size_t)rows_cache * d_model);
    dGamma.zero();  // kernel accumulates via atomicAdd across rows
    launch_rmsnorm_backward(dOut, x_cache, gamma.data, rms_cache.data, dX.data, dGamma.data, rows_cache, d_model);
    return dX.data;
}

void RMSNorm::update(float lr) {
    launch_sgd_update(gamma.data, dGamma.data, lr, (int)gamma.size);
}

// ============================================================================
// SwiGLU
// ============================================================================
SwiGLU::SwiGLU(int d_ff, float beta) : d_ff(d_ff), beta(beta) {
    dBetaAccum.alloc(1);
}

float* SwiGLU::forward(const float* x, int rows) {
    out.ensure((size_t)rows * d_ff);
    sig_cache.ensure((size_t)rows * d_ff);
    launch_swiglu_forward(x, beta, out.data, sig_cache.data, rows, d_ff);
    x_cache = x;
    rows_cache = rows;
    return out.data;
}

float* SwiGLU::backward(const float* dOut) {
    dX.ensure((size_t)rows_cache * 2 * d_ff);
    dBetaAccum.zero();
    launch_swiglu_backward(dOut, x_cache, sig_cache.data, beta, dX.data, dBetaAccum.data, rows_cache, d_ff);
    return dX.data;
}

void SwiGLU::update(float lr) {
    std::vector<float> h = dBetaAccum.to_host();
    beta -= lr * h[0];
}

// ============================================================================
// MLP
// ============================================================================
MLP::MLP(int d_model, int d_ff, std::mt19937& rng)
    : hidden_layer(d_model, 2 * d_ff, true, 0.1f, rng),
      swiglu(d_ff, 1.0f),
      output_layer(d_ff, d_model, true, 0.1f, rng) {}

float* MLP::forward(cublasHandle_t handle, const float* x, int rows) {
    float* h = hidden_layer.forward(handle, x, rows);
    float* s = swiglu.forward(h, rows);
    return output_layer.forward(handle, s, rows);
}

float* MLP::backward(cublasHandle_t handle, const float* dOut) {
    float* g = output_layer.backward(handle, dOut);
    g = swiglu.backward(g);
    return hidden_layer.backward(handle, g);
}

void MLP::update(float lr) {
    hidden_layer.update(lr);
    swiglu.update(lr);
    output_layer.update(lr);
}

// ============================================================================
// MultiHeadAttention
// ============================================================================
MultiHeadAttention::MultiHeadAttention(int d_model, int num_heads, std::mt19937& rng)
    : d_model(d_model), num_heads(num_heads), d_head(d_model / num_heads),
      Wq(d_model, d_model, false, 0.1f, rng),
      Wk(d_model, d_model, false, 0.1f, rng),
      Wv(d_model, d_model, false, 0.1f, rng),
      Wo(d_model, d_model, false, 0.1f, rng) {}

float* MultiHeadAttention::forward(cublasHandle_t handle, const float* x, int B, int T) {
    int rows = B * T;
    int H = num_heads, dh = d_head;
    size_t head_elems = (size_t)B * H * T * dh;

    float* Qflat = Wq.forward(handle, x, rows);
    float* Kflat = Wk.forward(handle, x, rows);
    float* Vflat = Wv.forward(handle, x, rows);

    Qh.ensure(head_elems);
    Kh.ensure(head_elems);
    Vh.ensure(head_elems);
    launch_split_heads(Qflat, Qh.data, B, T, H, dh);
    launch_split_heads(Kflat, Kh.data, B, T, H, dh);
    launch_split_heads(Vflat, Vh.data, B, T, H, dh);

    int batch = B * H;
    size_t score_elems = (size_t)batch * T * T;
    scores.ensure(score_elems);
    probs.ensure(score_elems);

    float scale = 1.0f / sqrtf((float)dh);
    gemm_batched(handle, Trans::N, Trans::T, T, T, dh, scale,
                 Qh.data, (long long)T * dh, Kh.data, (long long)T * dh,
                 0.0f, scores.data, (long long)T * T, batch, /*fp8=*/true);

    launch_causal_mask_add(scores.data, batch, T);
    launch_softmax_forward(scores.data, probs.data, batch * T, T);

    attn_out.ensure(head_elems);
    gemm_batched(handle, Trans::N, Trans::N, T, dh, T, 1.0f,
                 probs.data, (long long)T * T, Vh.data, (long long)T * dh,
                 0.0f, attn_out.data, (long long)T * dh, batch, /*fp8=*/true);

    merged.ensure((size_t)rows * d_model);
    launch_merge_heads(attn_out.data, merged.data, B, T, H, dh);

    B_cache = B;
    T_cache = T;
    return Wo.forward(handle, merged.data, rows);
}

float* MultiHeadAttention::backward(cublasHandle_t handle, const float* dOut) {
    int B = B_cache, T = T_cache;
    int rows = B * T;
    int H = num_heads, dh = d_head;
    int batch = B * H;
    size_t head_elems = (size_t)batch * T * dh;

    float* dMergedPtr = Wo.backward(handle, dOut);  // (rows, D)

    dAttnOut.ensure(head_elems);
    launch_split_heads(dMergedPtr, dAttnOut.data, B, T, H, dh);  // dL/d(attn_out heads)

    // dV = probs^T @ dOh
    dVh.ensure(head_elems);
    gemm_batched(handle, Trans::T, Trans::N, T, dh, T, 1.0f,
                 probs.data, (long long)T * T, dAttnOut.data, (long long)T * dh,
                 0.0f, dVh.data, (long long)T * dh, batch, /*fp8=*/true);

    // dAttn = dOh @ V^T  (stored temporarily in dScores, then softmax-backward'd in place)
    dScores.ensure((size_t)batch * T * T);
    gemm_batched(handle, Trans::N, Trans::T, T, T, dh, 1.0f,
                 dAttnOut.data, (long long)T * dh, Vh.data, (long long)T * dh,
                 0.0f, dScores.data, (long long)T * T, batch, /*fp8=*/true);
    launch_softmax_backward(dScores.data, probs.data, dScores.data, batch * T, T);

    float scale = 1.0f / sqrtf((float)dh);
    dQh.ensure(head_elems);
    gemm_batched(handle, Trans::N, Trans::N, T, dh, T, scale,
                 dScores.data, (long long)T * T, Kh.data, (long long)T * dh,
                 0.0f, dQh.data, (long long)T * dh, batch, /*fp8=*/true);

    dKh.ensure(head_elems);
    gemm_batched(handle, Trans::T, Trans::N, T, dh, T, scale,
                 dScores.data, (long long)T * T, Qh.data, (long long)T * dh,
                 0.0f, dKh.data, (long long)T * dh, batch, /*fp8=*/true);

    dMerged.ensure((size_t)rows * d_model);
    dx.ensure((size_t)rows * d_model);

    launch_merge_heads(dQh.data, dMerged.data, B, T, H, dh);
    float* dQm = Wq.backward(handle, dMerged.data);

    launch_merge_heads(dKh.data, dMerged.data, B, T, H, dh);
    float* dKm = Wk.backward(handle, dMerged.data);

    launch_merge_heads(dVh.data, dMerged.data, B, T, H, dh);
    float* dVm = Wv.backward(handle, dMerged.data);

    launch_residual_add(dx.data, dQm, dKm, rows * d_model);
    launch_residual_add(dx.data, dx.data, dVm, rows * d_model);
    return dx.data;
}

void MultiHeadAttention::update(float lr) {
    Wq.update(lr);
    Wk.update(lr);
    Wv.update(lr);
    Wo.update(lr);
}

// ============================================================================
// TransformerBlock
// ============================================================================
TransformerBlock::TransformerBlock(int d_model, int num_heads, int d_ff, std::mt19937& rng)
    : ln1(d_model), ln2(d_model), attn(d_model, num_heads, rng), mlp(d_model, d_ff, rng) {}

float* TransformerBlock::forward(cublasHandle_t handle, const float* x, int B, int T) {
    int rows = B * T;
    int d_model = ln1.d_model;

    float* ln1_out = ln1.forward(x, rows);
    float* attn_out = attn.forward(handle, ln1_out, B, T);
    x1.ensure((size_t)rows * d_model);
    launch_residual_add(x1.data, x, attn_out, rows * d_model);

    float* ln2_out = ln2.forward(x1.data, rows);
    float* mlp_out = mlp.forward(handle, ln2_out, rows);
    x2.ensure((size_t)rows * d_model);
    launch_residual_add(x2.data, x1.data, mlp_out, rows * d_model);

    B_cache = B;
    T_cache = T;
    return x2.data;
}

float* TransformerBlock::backward(cublasHandle_t handle, const float* dOut) {
    int rows = B_cache * T_cache;
    int d_model = ln1.d_model;

    float* dln2_out = mlp.backward(handle, dOut);
    float* dx1_from_mlp = ln2.backward(dln2_out);
    dx1.ensure((size_t)rows * d_model);
    launch_residual_add(dx1.data, dOut, dx1_from_mlp, rows * d_model);

    float* dln1_out = attn.backward(handle, dx1.data);
    float* dx0_from_attn = ln1.backward(dln1_out);
    dx0.ensure((size_t)rows * d_model);
    launch_residual_add(dx0.data, dx1.data, dx0_from_attn, rows * d_model);

    return dx0.data;
}

void TransformerBlock::update(float lr) {
    attn.update(lr);
    ln1.update(lr);
    mlp.update(lr);
    ln2.update(lr);
}

// ============================================================================
// Embedding
// ============================================================================
Embedding::Embedding(int vocab_size, int d_model, int max_len, std::mt19937& rng)
    : vocab_size(vocab_size), d_model(d_model), max_len(max_len) {
    token_emb.randn_init((size_t)vocab_size * d_model, 0.02f, rng);
    pos_emb.randn_init((size_t)max_len * d_model, 0.02f, rng);
}

float* Embedding::forward(const int* ids, int B, int T) {
    out.ensure((size_t)B * T * d_model);
    launch_embedding_forward(ids, token_emb.data, pos_emb.data, out.data, B, T, d_model);
    ids_cache = ids;
    B_cache = B;
    T_cache = T;
    return out.data;
}

void Embedding::backward(const float* dOut) {
    d_token_emb.ensure(token_emb.size);
    d_token_emb.zero();
    d_pos_emb.ensure(pos_emb.size);
    d_pos_emb.zero();
    launch_embedding_backward(ids_cache, dOut, d_token_emb.data, d_pos_emb.data, B_cache, T_cache, d_model);
}

void Embedding::update(float lr) {
    launch_sgd_update(token_emb.data, d_token_emb.data, lr, (int)token_emb.size);
    launch_sgd_update(pos_emb.data, d_pos_emb.data, lr, (int)pos_emb.size);
}

// ============================================================================
// TransformerModel
// ============================================================================
TransformerModel::TransformerModel(int vocab_size, int d_model, int num_heads, int num_layers,
                                    int d_ff, int max_len, std::mt19937& rng)
    : vocab_size(vocab_size), d_model(d_model), num_heads(num_heads), num_layers(num_layers),
      d_ff(d_ff), max_len(max_len),
      embedding(vocab_size, d_model, max_len, rng),
      ln_f(d_model),
      W_out(d_model, vocab_size, false, 0.02f, rng) {
    blocks.reserve(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        blocks.push_back(std::make_unique<TransformerBlock>(d_model, num_heads, d_ff, rng));
    }
}

float* TransformerModel::forward(cublasHandle_t handle, const int* ids, int B, int T) {
    float* x = embedding.forward(ids, B, T);
    for (auto& block : blocks) x = block->forward(handle, x, B, T);
    float* lnf_out = ln_f.forward(x, B * T);
    return W_out.forward(handle, lnf_out, B * T);
}

void TransformerModel::backward(cublasHandle_t handle, const float* dLogits) {
    float* dx = W_out.backward(handle, dLogits);
    dx = ln_f.backward(dx);
    for (auto it = blocks.rbegin(); it != blocks.rend(); ++it) dx = (*it)->backward(handle, dx);
    embedding.backward(dx);
}

void TransformerModel::update(float lr) {
    embedding.update(lr);
    for (auto& block : blocks) block->update(lr);
    ln_f.update(lr);
    W_out.update(lr);
}

// ============================================================================
// CrossEntropyLoss
// ============================================================================
CrossEntropyLoss::CrossEntropyLoss(int ignore_index) : ignore_index(ignore_index) {
    loss_sum_d.alloc(1);
    n_valid_d.alloc(1);
}

float CrossEntropyLoss::forward(const float* logits, const int* targets, int rows, int V) {
    probs.ensure((size_t)rows * V);
    loss_sum_d.zero();
    n_valid_d.zero();

    launch_cross_entropy_forward(logits, targets, probs.data, loss_sum_d.data, n_valid_d.data, rows, V, ignore_index);

    std::vector<float> h_loss = loss_sum_d.to_host();
    std::vector<int> h_nvalid = n_valid_d.to_host();

    targets_cache = targets;
    rows_cache = rows;
    V_cache = V;
    n_valid_cache = h_nvalid[0];

    float n = std::max(h_nvalid[0], 1);
    return h_loss[0] / n;
}

float* CrossEntropyLoss::backward() {
    dLogits.ensure((size_t)rows_cache * V_cache);
    launch_cross_entropy_backward(probs.data, targets_cache, dLogits.data, n_valid_cache, rows_cache, V_cache, ignore_index);
    return dLogits.data;
}
