// Persistent GPU inference server for the showcase chat UI.
//
// serve.py's default chat path runs generate_stream() (utils/generate.py) on
// the pure-NumPy model.py port, entirely on CPU - no KV cache, so every new
// token re-runs a full forward pass over the whole context in plain NumPy.
// That's fine for the training/eval code path but far too slow for a live
// chat UI. This binary loads the same checkpoint into the CUDA/cuBLAS
// TransformerModel (cuda/include/layers.cuh - the one main.cu trains with)
// and does the same autoregressive sampling loop on the GPU instead, so a
// forward pass costs a cuBLAS matmul rather than a numpy one.
//
// It is deliberately a long-lived process rather than one-shot like probe.cu:
// spawning a fresh process per chat message would pay CUDA context + cuBLAS
// handle + checkpoint-load setup cost (order of a second) on every turn.
// Instead serve.py starts one of these at startup and feeds it requests over
// stdin/stdout, line-delimited JSON in both directions - one already-rendered
// chat prompt (render_chat_prompt in utils/generate.py stays the source of
// truth for prompt formatting; this binary only samples).
//
// Protocol (newline-delimited JSON, UTF-8):
//   startup  -> stdout: {"ready":true,"step":N,"vocab_size":N,"max_len":N}
//   request  <- stdin:  {"prompt":"...","max_new_tokens":64,"temperature":0.8,
//                        "top_k":40,"top_p":0.95,"seed":null}
//   response -> stdout: {"token":"..."}                 (zero or more)
//                       {"done":true,"tokens":N,"elapsed":S,"tokens_per_sec":R}
//                    or {"error":"..."}
//
// Sampling (temperature/top_k/top_p) and the stop-string / held-back-suffix
// logic mirror utils/generate.py's sample_from_logits / generate_stream
// exactly, so switching engines doesn't change generation behavior.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "checkpoint.cuh"
#include "data.hpp"
#include "layers.cuh"

// ---------------------------------------------------------------------------
// Minimal JSON I/O: parsing requests with the existing minijson parser
// (data.hpp), and writing responses by hand (only strings/bools/numbers,
// so a full writer would be overkill - just escape and print).
// ---------------------------------------------------------------------------
static void json_write_escaped(std::ostream& os, const std::string& s) {
    os << '"';
    for (unsigned char c : s) {
        switch (c) {
            case '"': os << "\\\""; break;
            case '\\': os << "\\\\"; break;
            case '\n': os << "\\n"; break;
            case '\r': os << "\\r"; break;
            case '\t': os << "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x", c);
                    os << buf;
                } else {
                    os << (char)c;
                }
        }
    }
    os << '"';
}

struct Request {
    std::string prompt;
    int max_new_tokens = 64;
    double temperature = 0.8;
    int top_k = 40;
    double top_p = 0.95;
    double repetition_penalty = 1.0;
    bool has_seed = false;
    unsigned long long seed = 0;
};

static Request parse_request(const std::string& line) {
    minijson::JsonValue v = minijson::Parser(line).parse();
    Request r;
    if (auto* p = v.get("prompt")) r.prompt = p->str;
    if (auto* p = v.get("max_new_tokens")) r.max_new_tokens = (int)p->num;
    if (auto* p = v.get("temperature")) r.temperature = p->num;
    if (auto* p = v.get("top_k")) r.top_k = (int)p->num;
    if (auto* p = v.get("top_p")) r.top_p = p->num;
    if (auto* p = v.get("repetition_penalty")) r.repetition_penalty = p->num;
    if (auto* p = v.get("seed")) {
        if (p->type == minijson::JsonValue::Type::Num) {
            r.has_seed = true;
            r.seed = (unsigned long long)p->num;
        }
    }
    return r;
}

// ---------------------------------------------------------------------------
// Lenient UTF-8 decode of raw token bytes, mirroring Python's
// bytes.decode("utf-8", errors="replace"): invalid or truncated sequences
// become U+FFFD rather than raising or emitting garbage. Needed because the
// bbpe tokenizer's decode() just concatenates each token's raw bytes
// (tokenizer/src/bpe_model.cpp), same as the Python side.
// ---------------------------------------------------------------------------
static std::string utf8_lenient_decode(const std::string& bytes) {
    static const std::string REPLACEMENT = "\xEF\xBF\xBD";  // U+FFFD
    std::string out;
    out.reserve(bytes.size());
    size_t i = 0;
    while (i < bytes.size()) {
        unsigned char c0 = (unsigned char)bytes[i];
        int len;
        if (c0 < 0x80) len = 1;
        else if ((c0 & 0xE0) == 0xC0) len = 2;
        else if ((c0 & 0xF0) == 0xE0) len = 3;
        else if ((c0 & 0xF8) == 0xF0) len = 4;
        else { out += REPLACEMENT; ++i; continue; }  // stray continuation/invalid byte

        if (i + (size_t)len > bytes.size()) {
            // Truncated multi-byte sequence at the end of the buffer - hold it
            // back as a single replacement char rather than emitting raw bytes.
            out += REPLACEMENT;
            break;
        }
        bool ok = true;
        for (int k = 1; k < len; ++k) {
            unsigned char c = (unsigned char)bytes[i + k];
            if ((c & 0xC0) != 0x80) { ok = false; break; }
        }
        if (!ok) { out += REPLACEMENT; ++i; continue; }
        out.append(bytes, i, (size_t)len);
        i += (size_t)len;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Sampling: port of utils/generate.py::sample_from_logits.
// ---------------------------------------------------------------------------
// repetition_penalty follows Keskar et al.'s CTRL formulation (mirrors
// utils/generate.py::sample_from_logits): every id in seen_ids gets divided
// (if positive) or multiplied (if negative) by the penalty, applied before
// temperature/top_k/top_p so it reshapes the distribution those act on.
static void apply_repetition_penalty(std::vector<double>& l, double repetition_penalty,
                                      const std::vector<int>& seen_ids) {
    if (repetition_penalty == 1.0) return;
    for (int id : seen_ids) {
        l[id] = l[id] > 0 ? l[id] / repetition_penalty : l[id] * repetition_penalty;
    }
}

static int sample_from_logits(std::vector<float>& logits, double temperature,
                               int top_k, double top_p, std::mt19937_64& rng,
                               double repetition_penalty, const std::vector<int>& seen_ids) {
    int V = (int)logits.size();
    if (temperature <= 0.0) {
        std::vector<double> l(V);
        for (int i = 0; i < V; ++i) l[i] = (double)logits[i];
        apply_repetition_penalty(l, repetition_penalty, seen_ids);
        int best = 0;
        for (int i = 1; i < V; ++i) if (l[i] > l[best]) best = i;
        return best;
    }

    std::vector<double> l(V);
    for (int i = 0; i < V; ++i) l[i] = (double)logits[i];
    apply_repetition_penalty(l, repetition_penalty, seen_ids);
    for (int i = 0; i < V; ++i) l[i] /= temperature;

    if (top_k > 0 && top_k < V) {
        std::vector<int> idx(V);
        for (int i = 0; i < V; ++i) idx[i] = i;
        std::nth_element(idx.begin(), idx.end() - top_k, idx.end(),
                          [&](int a, int b) { return l[a] < l[b]; });
        std::vector<char> keep(V, 0);
        for (int i = V - top_k; i < V; ++i) keep[idx[i]] = 1;
        for (int i = 0; i < V; ++i) if (!keep[i]) l[i] = -std::numeric_limits<double>::infinity();
    }

    double maxv = -std::numeric_limits<double>::infinity();
    for (double x : l) maxv = std::max(maxv, x);
    std::vector<double> probs(V);
    double sum = 0.0;
    for (int i = 0; i < V; ++i) {
        probs[i] = std::isinf(l[i]) ? 0.0 : std::exp(l[i] - maxv);
        sum += probs[i];
    }
    for (double& p : probs) p /= sum;

    if (top_p > 0.0 && top_p < 1.0) {
        std::vector<int> order(V);
        for (int i = 0; i < V; ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](int a, int b) { return probs[a] > probs[b]; });
        double cumulative = 0.0;
        size_t cutoff = 0;
        for (; cutoff < order.size(); ++cutoff) {
            cumulative += probs[order[cutoff]];
            if (cumulative > top_p) { ++cutoff; break; }
        }
        if (cutoff == 0) cutoff = 1;
        std::vector<double> masked(V, 0.0);
        double msum = 0.0;
        for (size_t k = 0; k < cutoff; ++k) { masked[order[k]] = probs[order[k]]; msum += probs[order[k]]; }
        for (double& p : masked) p /= msum;
        probs = masked;
    }

    std::uniform_real_distribution<double> uni(0.0, 1.0);
    double r = uni(rng);
    double acc = 0.0;
    for (int i = 0; i < V; ++i) {
        acc += probs[i];
        if (r <= acc) return i;
    }
    return V - 1;
}

// Mirrors utils/generate.py::_held_back: characters that can't safely be
// emitted yet because they might be a truncated stop tag or a lone U+FFFD.
static size_t held_back(const std::string& text, const std::vector<std::string>& stops) {
    size_t hold = 0;
    const std::string FFFD = "\xEF\xBF\xBD";
    while (hold + FFFD.size() <= text.size() &&
           text.compare(text.size() - hold - FFFD.size(), FFFD.size(), FFFD) == 0) {
        hold += FFFD.size();
    }
    for (const auto& stop : stops) {
        for (size_t k = 1; k < stop.size() && k < text.size(); ++k) {
            if (text.size() >= k && text.compare(text.size() - k, k, stop, 0, k) == 0)
                hold = std::max(hold, k);
        }
    }
    return hold;
}

int main(int argc, char** argv) {
    std::string ckpt_path, tok_kind = "bbpe", tok_path;
    int requested_vocab = 32000;
    for (int i = 1; i < argc; ++i) {
        std::string flag = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", name); exit(1); }
            return argv[++i];
        };
        if (flag == "--ckpt") ckpt_path = next("--ckpt");
        else if (flag == "--tokenizer") tok_kind = next("--tokenizer");
        else if (flag == "--tokenizer-path") tok_path = next("--tokenizer-path");
        else if (flag == "--vocab-size") requested_vocab = std::stoi(next("--vocab-size"));
        else { fprintf(stderr, "unknown flag %s\n", flag.c_str()); exit(1); }
    }
    if (ckpt_path.empty()) { fprintf(stderr, "usage: %s --ckpt path [--tokenizer bbpe|byte] [--tokenizer-path path] [--vocab-size N]\n", argv[0]); return 1; }

    CheckpointHeader h = read_checkpoint_header(ckpt_path);
    DataTokenizer tokenizer = make_tokenizer(tok_kind, tok_path, requested_vocab);

    g_training = false;
    std::mt19937 init_rng(0);
    TransformerModel model(h.vocab_size, h.d_model, h.num_heads, h.num_layers, h.d_ff, h.max_len, init_rng);
    load_checkpoint(ckpt_path, model);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // "<|endoftext|>" (utils/chat.py::EOS_TAG) is a real single-token id -
    // see tokenizer/tools/remap_specials.cpp - appended after every
    // assistant turn during SFT (cuda/include/data.hpp's eos_tag()), so a
    // model trained on the current data learns to end its turn with one
    // token instead of spelling out the next role tag as literal text. The
    // role tags stay listed as a fallback for checkpoints trained before
    // EOS_TAG existed.
    const std::vector<std::string> stop_strings = {
        "<|endoftext|>", "<|user|>", "<|system|>", "<|assistant|>"};

    // If a stop string is registered as a single special token (see
    // tokenizer/tools/remap_specials.cpp), decode() with skip_special_tokens=true
    // below yields "" for it rather than the tag text, so the decoded-substring
    // search a few lines down would never see it. Stop on the sampled id
    // directly for any stop string that tokenizes to exactly one id; the
    // substring search remains the only mechanism for stop strings that
    // aren't (yet) registered as specials, e.g. the byte tokenizer never has any.
    std::vector<int> stop_token_ids;
    for (const auto& s : stop_strings) {
        std::vector<int> enc = tokenizer.encode(s);
        if (enc.size() == 1) stop_token_ids.push_back(enc[0]);
    }

    std::cout << "{\"ready\":true,\"step\":" << h.step
              << ",\"vocab_size\":" << h.vocab_size
              << ",\"max_len\":" << h.max_len << "}" << std::endl;

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        Request req;
        try {
            req = parse_request(line);
        } catch (const std::exception& e) {
            std::cout << "{\"error\":"; json_write_escaped(std::cout, std::string("bad request: ") + e.what());
            std::cout << "}" << std::endl;
            continue;
        }

        try {
            auto t0 = std::chrono::steady_clock::now();
            std::mt19937_64 rng(req.has_seed ? req.seed : std::random_device{}());

            std::vector<int> ids = tokenizer.encode(req.prompt);
            std::vector<int> generated;
            std::string decoded_emitted_prefix;  // bytes already yielded, for computing the new suffix
            size_t emitted = 0;
            int n = 0;
            bool stopped = false;

            for (; n < req.max_new_tokens; ++n) {
                int T = (int)ids.size();
                int start = std::max(0, T - h.max_len);
                std::vector<int> context(ids.begin() + start, ids.end());
                int ctx_len = (int)context.size();

                DeviceIntArray ids_dev;
                ids_dev.from_host(context);
                float* logits_dev = model.forward(handle, ids_dev.data, 1, ctx_len);

                int V = h.vocab_size;
                std::vector<float> last_logits(V);
                CUDA_CHECK(cudaMemcpy(last_logits.data(),
                                       logits_dev + (size_t)(ctx_len - 1) * V,
                                       V * sizeof(float), cudaMemcpyDeviceToHost));

                int next_id = sample_from_logits(last_logits, req.temperature, req.top_k, req.top_p, rng,
                                                  req.repetition_penalty, context);

                if (std::find(stop_token_ids.begin(), stop_token_ids.end(), next_id) != stop_token_ids.end()) {
                    std::string decoded = utf8_lenient_decode(tokenizer.decode(generated));
                    if (decoded.size() > emitted) {
                        std::cout << "{\"token\":";
                        json_write_escaped(std::cout, decoded.substr(emitted));
                        std::cout << "}" << std::endl;
                    }
                    stopped = true;
                    break;
                }

                ids.push_back(next_id);
                generated.push_back(next_id);

                std::string decoded = utf8_lenient_decode(tokenizer.decode(generated));

                size_t stop_at = std::string::npos;
                for (const auto& s : stop_strings) {
                    size_t pos = decoded.find(s);
                    if (pos != std::string::npos && pos < stop_at) stop_at = pos;
                }
                if (stop_at != std::string::npos) {
                    if (stop_at > emitted) {
                        std::cout << "{\"token\":";
                        json_write_escaped(std::cout, decoded.substr(emitted, stop_at - emitted));
                        std::cout << "}" << std::endl;
                    }
                    stopped = true;
                    break;
                }

                size_t safe = decoded.size() - held_back(decoded, stop_strings);
                if (safe > emitted) {
                    std::cout << "{\"token\":";
                    json_write_escaped(std::cout, decoded.substr(emitted, safe - emitted));
                    std::cout << "}" << std::endl;
                    emitted = safe;
                }
            }

            if (!stopped) {
                std::string decoded = utf8_lenient_decode(tokenizer.decode(generated));
                if (decoded.size() > emitted) {
                    std::cout << "{\"token\":";
                    json_write_escaped(std::cout, decoded.substr(emitted));
                    std::cout << "}" << std::endl;
                }
            }

            double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            double rate = elapsed > 0 ? n / elapsed : 0.0;
            std::cout << "{\"done\":true,\"tokens\":" << n
                      << ",\"elapsed\":" << elapsed
                      << ",\"tokens_per_sec\":" << rate << "}" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "{\"error\":"; json_write_escaped(std::cout, e.what());
            std::cout << "}" << std::endl;
        }
    }
    return 0;
}
