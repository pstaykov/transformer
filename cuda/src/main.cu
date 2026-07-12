// train_transformer_cuda
//
// CUDA/cuBLAS port of transformer/train.py + model.py (decoder-only
// transformer, RMSNorm + SwiGLU blocks, next-token prediction). Feature parity
// with the Python trainer: byte-level or bbpe tokenizer, plain-text or
// conversation-format (SFT) data with assistant-turn loss masking, optional
// simulated FP8 (E4M3) matmuls, checkpoint save/resume, and metrics.csv logging.
//
// Usage:
//   ./train_transformer_cuda [--corpus path] [--data-format text|chat]
//       [--max-bytes N] [--tokenizer byte|bbpe] [--tokenizer-path path] [--vocab-size N]
//       [--d-model N] [--num-heads N] [--num-layers N] [--d-ff N] [--seq-len N]
//       [--batch-size N] [--steps N] [--lr F] [--seed N] [--fp8]
//       [--log-every N] [--checkpoint-every N] [--checkpoint-dir dir]
//       [--metrics-path path] [--resume path] [--reset-step]
//
// Supervised fine-tuning (SFT) is just a resume onto conversation data. The
// architecture is read back from the checkpoint, so only the new data / lr /
// steps need to be given:
//
//   # 1. pretrain on plain text
//   ./train_transformer_cuda --corpus corpus.txt --tokenizer bbpe --steps 2000
//
//   # 2. fine-tune on chat data (same tokenizer, lower lr, its own outputs)
//   ./train_transformer_cuda --resume checkpoints/latest.ckpt \
//       --corpus conversations.json --data-format chat --tokenizer bbpe \
//       --lr 0.01 --steps 300 --reset-step \
//       --checkpoint-dir sft_ckpts --metrics-path sft_metrics.csv
//
// Only assistant-turn tokens are trained on (user/system turns are masked out
// of the loss); see utils/chat.py and data.hpp for the conversation format.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>

#include "common.cuh"
#include "layers.cuh"
#include "checkpoint.cuh"
#include "data.hpp"

// Target value excluded from the loss (padding / non-assistant tokens). Matches
// utils/chat.py::IGNORE_INDEX and the CrossEntropyLoss ignore_index below.
static const int IGNORE_INDEX = -100;

struct Args {
    std::string corpus = "../data/tiny_corpus.txt";
    std::string data_format = "text";     // text | chat
    std::string tokenizer = "byte";       // byte | bbpe
    std::string tokenizer_path = "../tokenizer/tok_out/tokenizer.bbpe";
    int vocab_size = 32000;               // only used with --tokenizer bbpe
    size_t max_bytes = 0;                 // cap on how much of --corpus is read (text format only); 0 = unlimited

    int d_model = 512;
    int num_heads = 8;
    int num_layers = 24;
    int d_ff = 2026;
    int seq_len = 32;
    int batch_size = 8;
    int steps = 500;
    float lr = 0.05f;
    int seed = 0;
    bool fp8 = false;

    int log_every = 10;
    int checkpoint_every = 100;
    std::string checkpoint_dir = "checkpoints";
    std::string metrics_path = "metrics.csv";
    std::string resume;
    bool reset_step = false;
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string flag = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", name);
                exit(EXIT_FAILURE);
            }
            return argv[++i];
        };
        if (flag == "--corpus") a.corpus = next("--corpus");
        else if (flag == "--data-format") a.data_format = next("--data-format");
        else if (flag == "--tokenizer") a.tokenizer = next("--tokenizer");
        else if (flag == "--tokenizer-path") a.tokenizer_path = next("--tokenizer-path");
        else if (flag == "--vocab-size") a.vocab_size = std::stoi(next("--vocab-size"));
        else if (flag == "--max-bytes") a.max_bytes = (size_t)std::stoull(next("--max-bytes"));
        else if (flag == "--d-model") a.d_model = std::stoi(next("--d-model"));
        else if (flag == "--num-heads") a.num_heads = std::stoi(next("--num-heads"));
        else if (flag == "--num-layers") a.num_layers = std::stoi(next("--num-layers"));
        else if (flag == "--d-ff") a.d_ff = std::stoi(next("--d-ff"));
        else if (flag == "--seq-len") a.seq_len = std::stoi(next("--seq-len"));
        else if (flag == "--batch-size") a.batch_size = std::stoi(next("--batch-size"));
        else if (flag == "--steps") a.steps = std::stoi(next("--steps"));
        else if (flag == "--lr") a.lr = std::stof(next("--lr"));
        else if (flag == "--seed") a.seed = std::stoi(next("--seed"));
        else if (flag == "--fp8") a.fp8 = true;
        else if (flag == "--log-every") a.log_every = std::stoi(next("--log-every"));
        else if (flag == "--checkpoint-every") a.checkpoint_every = std::stoi(next("--checkpoint-every"));
        else if (flag == "--checkpoint-dir") a.checkpoint_dir = next("--checkpoint-dir");
        else if (flag == "--metrics-path") a.metrics_path = next("--metrics-path");
        else if (flag == "--resume") a.resume = next("--resume");
        else if (flag == "--reset-step") a.reset_step = true;
        else fprintf(stderr, "[WARN] unknown argument: %s\n", flag.c_str());
    }
    return a;
}

// Random contiguous windows for next-token prediction: x = ids[s : s+T],
// y = ids[s+1 : s+T+1]. Targets whose predict-mask is false become IGNORE_INDEX
// so the loss skips them (user/system turns in chat data). Mirrors train.py's
// sample_batch.
static void sample_batch(const Dataset& data, int batch_size, int seq_len,
                          std::mt19937& rng, std::vector<int>& x_host, std::vector<int>& y_host) {
    const std::vector<int>& ids = data.ids;
    const std::vector<char>& mask = data.mask;
    int max_start = (int)ids.size() - seq_len - 1;  // start drawn from [0, max_start)
    std::uniform_int_distribution<int> dist(0, max_start - 1);
    x_host.resize((size_t)batch_size * seq_len);
    y_host.resize((size_t)batch_size * seq_len);
    for (int b = 0; b < batch_size; ++b) {
        int s = dist(rng);
        for (int t = 0; t < seq_len; ++t) {
            x_host[(size_t)b * seq_len + t] = ids[s + t];
            int tgt = ids[s + t + 1];
            y_host[(size_t)b * seq_len + t] = mask[s + t + 1] ? tgt : IGNORE_INDEX;
        }
    }
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    const bool resuming = !args.resume.empty();

    // When resuming (e.g. fine-tuning / SFT from a pretrained checkpoint), the
    // model architecture is taken from the checkpoint, so the arch flags don't
    // have to be repeated - just point at the new data with --lr/--steps/etc.
    if (resuming) {
        CheckpointHeader h = read_checkpoint_header(args.resume);
        args.d_model = h.d_model;
        args.num_heads = h.num_heads;
        args.num_layers = h.num_layers;
        args.d_ff = h.d_ff;
        args.seq_len = h.max_len;
        args.vocab_size = h.vocab_size;
        printf("Resuming from %s (saved at step %d): architecture taken from the checkpoint\n"
               "  d_model=%d num_heads=%d num_layers=%d d_ff=%d seq_len=%d vocab_size=%d\n"
               "  (any --d-model/--num-heads/--num-layers/--d-ff/--seq-len on the command line are ignored)\n",
               args.resume.c_str(), h.step, args.d_model, args.num_heads, args.num_layers,
               args.d_ff, args.seq_len, args.vocab_size);
    }

    DataTokenizer tokenizer = make_tokenizer(args.tokenizer, args.tokenizer_path, args.vocab_size);
    const int vocab_size = tokenizer.vocab_size();

    // The model's vocab is fixed by the checkpoint; the tokenizer must agree.
    if (resuming && vocab_size != args.vocab_size) {
        fprintf(stderr,
                "[error] checkpoint was trained with vocab_size=%d but tokenizer '%s' yields vocab_size=%d.\n"
                "        Fine-tune with the same tokenizer used for pretraining "
                "(e.g. --tokenizer bbpe --tokenizer-path <path>).\n",
                args.vocab_size, args.tokenizer.c_str(), vocab_size);
        return EXIT_FAILURE;
    }

    Dataset data = (args.data_format == "chat")
                       ? load_chat_dataset(tokenizer, args.corpus)
                       : load_text_dataset(tokenizer, args.corpus, args.max_bytes);

    if ((int)data.ids.size() <= args.seq_len + 1) {
        fprintf(stderr, "[error] corpus has only %zu tokens, need > seq_len+1 (%d)\n",
                data.ids.size(), args.seq_len + 1);
        return EXIT_FAILURE;
    }
    printf("Loaded corpus: %zu tokens, vocab_size=%d (tokenizer=%s, format=%s, fp8=%s)\n",
           data.ids.size(), vocab_size, args.tokenizer.c_str(), args.data_format.c_str(),
           args.fp8 ? "on" : "off");

    g_fp8_enabled = args.fp8;

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    std::mt19937 rng((unsigned int)args.seed);
    TransformerModel model(vocab_size, args.d_model, args.num_heads, args.num_layers,
                            args.d_ff, args.seq_len, rng);
    CrossEntropyLoss loss_fn(/*ignore_index=*/IGNORE_INDEX);

    int start_step = 0;
    if (resuming) {
        start_step = load_checkpoint(args.resume, model);
        if (args.reset_step) {
            printf("Loaded weights from %s; step counter reset to 0 for this run\n", args.resume.c_str());
            start_step = 0;
        } else {
            printf("Resumed weights from %s; continuing at step %d\n", args.resume.c_str(), start_step);
        }
    }

    printf("d_model=%d num_heads=%d num_layers=%d d_ff=%d seq_len=%d batch_size=%d\n",
           args.d_model, args.num_heads, args.num_layers, args.d_ff, args.seq_len, args.batch_size);

    ensure_dir(args.checkpoint_dir);

    // Append to metrics.csv, writing the header only for a fresh file.
    bool metrics_is_new = !std::ifstream(args.metrics_path).good();
    std::ofstream metrics(args.metrics_path, std::ios::app);
    if (metrics_is_new)
        metrics << "step,loss,perplexity,lr,tokens_per_sec,elapsed_sec\n";

    printf("Training for %d steps (starting at %d)...\n", args.steps, start_step);

    DeviceIntArray x_dev, y_dev;
    std::vector<int> x_host, y_host;

    auto t_start = std::chrono::steady_clock::now();
    int rows = args.batch_size * args.seq_len;
    float loss = 0.0f;
    int final_step = start_step + args.steps - 1;

    for (int step = start_step; step < start_step + args.steps; ++step) {
        auto t0 = std::chrono::steady_clock::now();

        sample_batch(data, args.batch_size, args.seq_len, rng, x_host, y_host);
        x_dev.from_host(x_host);
        y_dev.from_host(y_host);

        float* logits = model.forward(handle, x_dev.data, args.batch_size, args.seq_len);
        loss = loss_fn.forward(logits, y_dev.data, rows, vocab_size);
        float* dlogits = loss_fn.backward();
        model.backward(handle, dlogits);
        model.update(args.lr);

        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count();
        double tokens_per_sec = (args.batch_size * args.seq_len) / std::max(dt, 1e-9);
        double elapsed = std::chrono::duration<double>(t1 - t_start).count();
        double ppl = std::exp(std::min(loss, 20.0f));

        metrics << step << ',' << loss << ',' << ppl << ',' << args.lr << ','
                << tokens_per_sec << ',' << elapsed << '\n';
        metrics.flush();

        if (step % args.log_every == 0 || step == final_step) {
            printf("step %6d | loss %.4f | ppl %8.2f | %8.1f tok/s | elapsed %6.1fs\n",
                   step, loss, ppl, tokens_per_sec, elapsed);
        }

        if (step > start_step && step % args.checkpoint_every == 0) {
            std::string ckpt = args.checkpoint_dir + "/ckpt_step" + std::to_string(step) + ".ckpt";
            save_checkpoint(ckpt, model, step);
            save_checkpoint(args.checkpoint_dir + "/latest.ckpt", model, step);
            printf("  saved checkpoint -> %s\n", ckpt.c_str());
        }
    }

    std::string final_ckpt = args.checkpoint_dir + "/ckpt_step" + std::to_string(final_step) + ".ckpt";
    save_checkpoint(final_ckpt, model, final_step);
    save_checkpoint(args.checkpoint_dir + "/latest.ckpt", model, final_step);
    printf("Final checkpoint -> %s\n", final_ckpt.c_str());

    fp8_free_scratch();
    CUBLAS_CHECK(cublasDestroy(handle));
    return 0;
}
