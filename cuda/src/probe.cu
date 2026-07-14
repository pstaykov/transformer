// Debug probe: load a checkpoint, forward a fixed token sequence, dump logits
// to stdout so they can be diffed against the numpy port's forward pass.
#include <cstdio>
#include <random>
#include <vector>

#include "checkpoint.cuh"
#include "layers.cuh"

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <checkpoint.ckpt> <id0> <id1> ...\n", argv[0]);
        return 1;
    }
    std::string path = argv[1];
    std::vector<int> ids;
    for (int i = 2; i < argc; ++i) ids.push_back(std::atoi(argv[i]));

    CheckpointHeader h = read_checkpoint_header(path);
    fprintf(stderr, "step=%d vocab=%d d_model=%d heads=%d layers=%d d_ff=%d max_len=%d\n",
            h.step, h.vocab_size, h.d_model, h.num_heads, h.num_layers, h.d_ff, h.max_len);

    g_training = false;  // eval mode: dropout off
    std::mt19937 rng(0);
    TransformerModel model(h.vocab_size, h.d_model, h.num_heads, h.num_layers, h.d_ff, h.max_len, rng);
    load_checkpoint(path, model);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    int T = (int)ids.size();
    DeviceIntArray ids_dev;
    ids_dev.from_host(ids);

    float* logits_dev = model.forward(handle, ids_dev.data, 1, T);

    int V = h.vocab_size;
    std::vector<float> logits_host((size_t)T * V);
    CUDA_CHECK(cudaMemcpy(logits_host.data(), logits_dev, logits_host.size() * sizeof(float),
                           cudaMemcpyDeviceToHost));

    // Print last-position logits (first 20 values, full argmax + a checksum).
    const float* last = logits_host.data() + (size_t)(T - 1) * V;
    double sum = 0.0, sumsq = 0.0;
    int argmax = 0;
    float maxv = last[0];
    for (int v = 0; v < V; ++v) {
        sum += last[v];
        sumsq += (double)last[v] * last[v];
        if (last[v] > maxv) { maxv = last[v]; argmax = v; }
    }
    printf("last_pos_first20:");
    for (int v = 0; v < 20; ++v) printf(" %.6f", last[v]);
    printf("\n");
    printf("sum=%.6f sumsq=%.6f argmax=%d maxv=%.6f\n", sum, sumsq, argmax, maxv);
    return 0;
}
