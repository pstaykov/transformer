#pragma once
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <sys/stat.h>

#include "layers.cuh"

// Binary checkpointing for the CUDA trainer, the counterpart of
// utils/checkpoint.py. It's a self-contained little-endian format rather than
// numpy's .npz (a zip of .npy blobs) - the two trainers keep their own
// checkpoints, but the on-disk layout here mirrors what save_checkpoint stores:
// every trainable parameter, flattened to a named float buffer, plus the step
// count and enough architecture metadata to refuse a mismatched --resume.
//
//   magic "TFCKPT1\n" (8 bytes)
//   int32  step
//   int32  vocab_size, d_model, num_heads, num_layers, d_ff, max_len
//   int32  num_params
//   repeat num_params:
//       int32   name_len
//       char[]  name
//       int64   size            (element count)
//       float[] data            (size floats)
//
// SwiGLU's beta is a single host-side float (it's passed by value into the
// kernels, not stored on the device), so it rides along as a size-1 param with
// a host pointer instead of a device pointer.

// A trainable parameter: exactly one of dev/host is non-null.
struct NamedParam {
    std::string name;
    float* dev = nullptr;   // device buffer (DeviceArray-backed params)
    float* host = nullptr;  // host scalar (e.g. SwiGLU beta)
    size_t size = 0;
};

namespace ckpt_detail {

inline void push_linear(std::vector<NamedParam>& out, const std::string& prefix, Linear& lin) {
    out.push_back({prefix + ".W", lin.W.data, nullptr, lin.W.size});
    if (lin.has_bias)
        out.push_back({prefix + ".b", lin.b.data, nullptr, lin.b.size});
}

}  // namespace ckpt_detail

// Walk the model in a fixed, stable order and list every trainable parameter.
// The same traversal is used by save and load, so ordering + names line up.
inline std::vector<NamedParam> collect_params(TransformerModel& m) {
    std::vector<NamedParam> out;
    out.push_back({"embedding.token_emb", m.embedding.token_emb.data, nullptr, m.embedding.token_emb.size});
    out.push_back({"embedding.pos_emb", m.embedding.pos_emb.data, nullptr, m.embedding.pos_emb.size});

    for (size_t i = 0; i < m.blocks.size(); ++i) {
        TransformerBlock& blk = *m.blocks[i];
        std::string p = "blocks." + std::to_string(i);
        out.push_back({p + ".ln1.gamma", blk.ln1.gamma.data, nullptr, blk.ln1.gamma.size});
        out.push_back({p + ".ln2.gamma", blk.ln2.gamma.data, nullptr, blk.ln2.gamma.size});
        ckpt_detail::push_linear(out, p + ".attn.Wq", blk.attn.Wq);
        ckpt_detail::push_linear(out, p + ".attn.Wk", blk.attn.Wk);
        ckpt_detail::push_linear(out, p + ".attn.Wv", blk.attn.Wv);
        ckpt_detail::push_linear(out, p + ".attn.Wo", blk.attn.Wo);
        ckpt_detail::push_linear(out, p + ".mlp.hidden", blk.mlp.hidden_layer);
        out.push_back({p + ".mlp.swiglu.beta", nullptr, &blk.mlp.swiglu.beta, 1});
        ckpt_detail::push_linear(out, p + ".mlp.output", blk.mlp.output_layer);
    }

    out.push_back({"ln_f.gamma", m.ln_f.gamma.data, nullptr, m.ln_f.gamma.size});
    ckpt_detail::push_linear(out, "W_out", m.W_out);
    return out;
}

// Architecture + step metadata stored at the front of a checkpoint. Read this
// with read_checkpoint_header() to reconstruct a model that matches a saved one
// (used by --resume so the arch flags don't have to be repeated for SFT).
struct CheckpointHeader {
    int step = 0;
    int vocab_size = 0;
    int d_model = 0;
    int num_heads = 0;
    int num_layers = 0;
    int d_ff = 0;
    int max_len = 0;
};

namespace ckpt_detail {

inline const char MAGIC[8] = {'T', 'F', 'C', 'K', 'P', 'T', '1', '\n'};

template <typename T>
inline void write_pod(std::ostream& os, const T& v) {
    os.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

template <typename T>
inline bool read_pod(std::istream& is, T& v) {
    return static_cast<bool>(is.read(reinterpret_cast<char*>(&v), sizeof(T)));
}

}  // namespace ckpt_detail

// Read only the header (magic + step + architecture) from a checkpoint, without
// touching the parameter blobs. Aborts on a missing/invalid file.
inline CheckpointHeader read_checkpoint_header(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) {
        fprintf(stderr, "[checkpoint] cannot open %s for reading\n", path.c_str());
        exit(EXIT_FAILURE);
    }
    char magic[8];
    is.read(magic, sizeof(magic));
    if (!is || std::memcmp(magic, ckpt_detail::MAGIC, sizeof(magic)) != 0) {
        fprintf(stderr, "[checkpoint] %s is not a valid checkpoint (bad magic)\n", path.c_str());
        exit(EXIT_FAILURE);
    }
    CheckpointHeader h;
    ckpt_detail::read_pod(is, h.step);
    ckpt_detail::read_pod(is, h.vocab_size);
    ckpt_detail::read_pod(is, h.d_model);
    ckpt_detail::read_pod(is, h.num_heads);
    ckpt_detail::read_pod(is, h.num_layers);
    ckpt_detail::read_pod(is, h.d_ff);
    ckpt_detail::read_pod(is, h.max_len);
    if (!is) {
        fprintf(stderr, "[checkpoint] %s: truncated header\n", path.c_str());
        exit(EXIT_FAILURE);
    }
    return h;
}

// Save the model params + step to `path`. Aborts on any I/O error.
inline void save_checkpoint(const std::string& path, TransformerModel& m, int step) {
    std::ofstream os(path, std::ios::binary | std::ios::trunc);
    if (!os) {
        fprintf(stderr, "[checkpoint] cannot open %s for writing\n", path.c_str());
        exit(EXIT_FAILURE);
    }
    os.write(ckpt_detail::MAGIC, sizeof(ckpt_detail::MAGIC));
    ckpt_detail::write_pod<int32_t>(os, step);
    ckpt_detail::write_pod<int32_t>(os, m.vocab_size);
    ckpt_detail::write_pod<int32_t>(os, m.d_model);
    ckpt_detail::write_pod<int32_t>(os, m.num_heads);
    ckpt_detail::write_pod<int32_t>(os, m.num_layers);
    ckpt_detail::write_pod<int32_t>(os, m.d_ff);
    ckpt_detail::write_pod<int32_t>(os, m.max_len);

    std::vector<NamedParam> params = collect_params(m);
    ckpt_detail::write_pod<int32_t>(os, (int32_t)params.size());

    std::vector<float> host;
    for (const NamedParam& p : params) {
        ckpt_detail::write_pod<int32_t>(os, (int32_t)p.name.size());
        os.write(p.name.data(), p.name.size());
        ckpt_detail::write_pod<int64_t>(os, (int64_t)p.size);

        if (p.dev) {
            host.resize(p.size);
            CUDA_CHECK(cudaMemcpy(host.data(), p.dev, p.size * sizeof(float), cudaMemcpyDeviceToHost));
            os.write(reinterpret_cast<const char*>(host.data()), p.size * sizeof(float));
        } else {
            os.write(reinterpret_cast<const char*>(p.host), p.size * sizeof(float));
        }
    }
    if (!os) {
        fprintf(stderr, "[checkpoint] write failed for %s\n", path.c_str());
        exit(EXIT_FAILURE);
    }
}

// Load params from `path` into `m` in place. Validates the magic and that the
// architecture matches the model being resumed. Returns the saved step.
inline int load_checkpoint(const std::string& path, TransformerModel& m) {
    std::ifstream is(path, std::ios::binary);
    if (!is) {
        fprintf(stderr, "[checkpoint] cannot open %s for reading\n", path.c_str());
        exit(EXIT_FAILURE);
    }
    char magic[8];
    is.read(magic, sizeof(magic));
    if (!is || std::memcmp(magic, ckpt_detail::MAGIC, sizeof(magic)) != 0) {
        fprintf(stderr, "[checkpoint] %s is not a valid checkpoint (bad magic)\n", path.c_str());
        exit(EXIT_FAILURE);
    }

    int32_t step = 0, vocab_size = 0, d_model = 0, num_heads = 0, num_layers = 0, d_ff = 0, max_len = 0;
    ckpt_detail::read_pod(is, step);
    ckpt_detail::read_pod(is, vocab_size);
    ckpt_detail::read_pod(is, d_model);
    ckpt_detail::read_pod(is, num_heads);
    ckpt_detail::read_pod(is, num_layers);
    ckpt_detail::read_pod(is, d_ff);
    ckpt_detail::read_pod(is, max_len);

    if (vocab_size != m.vocab_size || d_model != m.d_model || num_heads != m.num_heads ||
        num_layers != m.num_layers || d_ff != m.d_ff || max_len != m.max_len) {
        fprintf(stderr,
                "[checkpoint] architecture mismatch resuming %s\n"
                "  checkpoint: vocab=%d d_model=%d heads=%d layers=%d d_ff=%d max_len=%d\n"
                "  model:      vocab=%d d_model=%d heads=%d layers=%d d_ff=%d max_len=%d\n",
                path.c_str(), vocab_size, d_model, num_heads, num_layers, d_ff, max_len,
                m.vocab_size, m.d_model, m.num_heads, m.num_layers, m.d_ff, m.max_len);
        exit(EXIT_FAILURE);
    }

    int32_t num_params = 0;
    ckpt_detail::read_pod(is, num_params);
    std::vector<NamedParam> params = collect_params(m);
    if ((size_t)num_params != params.size()) {
        fprintf(stderr, "[checkpoint] param count mismatch: file has %d, model has %zu\n",
                num_params, params.size());
        exit(EXIT_FAILURE);
    }

    std::vector<float> host;
    for (NamedParam& p : params) {
        int32_t name_len = 0;
        ckpt_detail::read_pod(is, name_len);
        std::string name(name_len, '\0');
        is.read(&name[0], name_len);
        int64_t size = 0;
        ckpt_detail::read_pod(is, size);

        if (!is || name != p.name || (size_t)size != p.size) {
            fprintf(stderr, "[checkpoint] param mismatch: expected %s (%zu), got %s (%lld)\n",
                    p.name.c_str(), p.size, name.c_str(), (long long)size);
            exit(EXIT_FAILURE);
        }

        host.resize(p.size);
        is.read(reinterpret_cast<char*>(host.data()), p.size * sizeof(float));
        if (!is) {
            fprintf(stderr, "[checkpoint] truncated reading %s from %s\n", p.name.c_str(), path.c_str());
            exit(EXIT_FAILURE);
        }
        if (p.dev)
            CUDA_CHECK(cudaMemcpy(p.dev, host.data(), p.size * sizeof(float), cudaMemcpyHostToDevice));
        else
            *p.host = host[0];
    }
    return step;
}

// mkdir -p for a single-level directory (matches os.makedirs(dir, exist_ok=True)
// usage in train.py, where checkpoint_dir is a plain relative dir).
inline void ensure_dir(const std::string& dir) {
    if (dir.empty() || dir == ".") return;
    if (mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "[checkpoint] could not create dir %s\n", dir.c_str());
        exit(EXIT_FAILURE);
    }
}
