#pragma once
#include <vector>
#include <random>
#include <cstring>
#include "common.cuh"

// A flat device float buffer. Shape is tracked for bookkeeping only - all ops
// treat it as a 1D array of `size` floats, sliced/reshaped by whoever uses it.
struct DeviceArray {
    float* data = nullptr;
    size_t size = 0;

    DeviceArray() = default;

    explicit DeviceArray(size_t n) { alloc(n); }

    void alloc(size_t n) {
        free();
        size = n;
        if (n > 0) CUDA_CHECK(cudaMalloc(&data, n * sizeof(float)));
    }

    void free() {
        if (data) {
            cudaFree(data);
            data = nullptr;
        }
        size = 0;
    }

    void zero() {
        if (size > 0) CUDA_CHECK(cudaMemset(data, 0, size * sizeof(float)));
    }

    // Reallocate only if the requested size actually changed, so layers can
    // call this on every forward() (batch/seq-len usually constant across
    // steps) without paying a cudaMalloc/cudaFree per step.
    void ensure(size_t n) {
        if (size != n) alloc(n);
    }

    void from_host(const std::vector<float>& host) {
        alloc(host.size());
        CUDA_CHECK(cudaMemcpy(data, host.data(), size * sizeof(float), cudaMemcpyHostToDevice));
    }

    std::vector<float> to_host() const {
        std::vector<float> host(size);
        if (size > 0) CUDA_CHECK(cudaMemcpy(host.data(), data, size * sizeof(float), cudaMemcpyDeviceToHost));
        return host;
    }

    // Kaiming/normal-ish init matching the numpy reference: randn(...) * scale
    void randn_init(size_t n, float scale, std::mt19937& rng) {
        std::normal_distribution<float> dist(0.0f, 1.0f);
        std::vector<float> host(n);
        for (auto& v : host) v = dist(rng) * scale;
        from_host(host);
    }

    void fill_init(size_t n, float value) {
        std::vector<float> host(n, value);
        from_host(host);
    }

    ~DeviceArray() { free(); }

    // Non-copyable (owns a GPU allocation), movable.
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    DeviceArray(DeviceArray&& o) noexcept : data(o.data), size(o.size) { o.data = nullptr; o.size = 0; }
    DeviceArray& operator=(DeviceArray&& o) noexcept {
        if (this != &o) { free(); data = o.data; size = o.size; o.data = nullptr; o.size = 0; }
        return *this;
    }
};

// Same as DeviceArray but for int32 buffers (token ids, targets, small
// scalar counters like cross-entropy's n_valid).
struct DeviceIntArray {
    int* data = nullptr;
    size_t size = 0;

    DeviceIntArray() = default;
    explicit DeviceIntArray(size_t n) { alloc(n); }

    void alloc(size_t n) {
        free();
        size = n;
        if (n > 0) CUDA_CHECK(cudaMalloc(&data, n * sizeof(int)));
    }

    void free() {
        if (data) {
            cudaFree(data);
            data = nullptr;
        }
        size = 0;
    }

    void ensure(size_t n) {
        if (size != n) alloc(n);
    }

    void zero() {
        if (size > 0) CUDA_CHECK(cudaMemset(data, 0, size * sizeof(int)));
    }

    void from_host(const std::vector<int>& host) {
        alloc(host.size());
        CUDA_CHECK(cudaMemcpy(data, host.data(), size * sizeof(int), cudaMemcpyHostToDevice));
    }

    std::vector<int> to_host() const {
        std::vector<int> host(size);
        if (size > 0) CUDA_CHECK(cudaMemcpy(host.data(), data, size * sizeof(int), cudaMemcpyDeviceToHost));
        return host;
    }

    ~DeviceIntArray() { free(); }

    DeviceIntArray(const DeviceIntArray&) = delete;
    DeviceIntArray& operator=(const DeviceIntArray&) = delete;
    DeviceIntArray(DeviceIntArray&& o) noexcept : data(o.data), size(o.size) { o.data = nullptr; o.size = 0; }
    DeviceIntArray& operator=(DeviceIntArray&& o) noexcept {
        if (this != &o) { free(); data = o.data; size = o.size; o.data = nullptr; o.size = 0; }
        return *this;
    }
};
