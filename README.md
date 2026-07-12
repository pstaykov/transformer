# transformer

A decoder-only transformer (LLaMA-style: RMSNorm + SwiGLU feed-forward, learned
positional embeddings) implemented **from scratch** with hand-written forward
and backward passes — no autograd framework. It ships in two independent
implementations that mirror each other 1:1:

| Implementation | Runs on | Speed | Use it for |
| --- | --- | --- | --- |
| **`train.py`** (+ `model.py`, `utils/`) | CPU, NumPy | slower | the reference; simulated FP8, chat SFT, resumable |
| **`cuda/`** (C++/CUDA + cuBLAS) | NVIDIA GPU | much faster | real training runs; same features, GPU kernels |

Both train with next-token prediction, log `step / loss / perplexity` to a
`metrics.csv`, checkpoint periodically, and support plain-text **or**
conversation-format (SFT) data with a byte-level or a real byte-pair-encoding
(bbpe) tokenizer.

---

## Repository layout

```
transformer/
├── train.py            # NumPy trainer (CPU)
├── model.py            # the decoder-only transformer (forward + backward)
├── utils/              # layers (attention, RMSNorm, SwiGLU, ...), fp8, losses,
│                       #   checkpoint I/O, chat/SFT data loading
├── cuda/               # standalone CUDA/cuBLAS trainer (its own README section)
│   ├── src/  include/  CMakeLists.txt
├── tokenizer/          # C++ byte-pair-encoding tokenizer + Python bindings
│   └── tok_out/        # a pretrained tokenizer (tokenizer.bbpe, vocab.json, ...)
└── data/
    ├── tiny_corpus.txt            # tiny plain-text sample
    └── sample_conversations.json  # tiny chat/SFT sample
```

---

## Quick start — NumPy trainer (CPU, zero build)

Requires **Python 3.12+**, **NumPy**, and **Numba** (Numba JIT-compiles the FP8
quantization kernel; it is imported at startup, so it's required even for a
quick run):

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy numba
```

Train on the bundled plain-text sample with the built-in byte tokenizer (no
tokenizer build needed — every UTF-8 byte is a token, `vocab_size = 257`):

```bash
python train.py \
    --corpus data/tiny_corpus.txt \
    --d-model 128 --num-heads 8 --num-layers 4 --d-ff 256 --seq-len 32 \
    --batch-size 8 --steps 200
```

> The defaults (`--d-model 512 --num-layers 24 --d-ff 2026`) describe a
> ~100M-parameter model, which is slow in pure NumPy on CPU. Use the smaller
> flags above to see it train quickly, and scale up from there.

You'll see per-step logs, a growing `metrics.csv`, and checkpoints written to
`checkpoints/` (`.npz`). Resume any run:

```bash
python train.py --resume checkpoints/latest.npz --steps 200
```

---

## Quick start — CUDA trainer (GPU, fast)

Requires an **NVIDIA GPU**, the **CUDA Toolkit** (tested with 12.x), **CMake ≥
3.18**, and a **C++17** host compiler.

```bash
cd cuda
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Run it from the `cuda/` directory so the default `--corpus ../data/...` paths
resolve (or pass `--corpus` explicitly):

```bash
./build/train_transformer_cuda \
    --corpus ../data/tiny_corpus.txt \
    --d-model 128 --num-heads 8 --num-layers 4 --d-ff 256 --seq-len 32 \
    --batch-size 8 --steps 500
```

The CUDA build links the tokenizer's C++ core automatically for `--tokenizer
bbpe` (`-DWITH_BBPE=ON`, the default). To build the byte-only trainer without
the tokenizer, configure with `-DWITH_BBPE=OFF`.

The CUDA trainer runs in **fp32** by default; add `--fp8` to enable the
simulated FP8 (E4M3) matmul path (matches `train.py`'s numerics, a bit slower).

---

## Data formats

**Plain text** (`--data-format text`, the default): the whole file is one token
stream; every position is a training target.

**Chat / SFT** (`--data-format chat`): a `.json` or `.jsonl` file of
conversations in the OpenAI message shape. Each conversation is rendered with
role tags and **only assistant-turn tokens are trained on** (user/system turns
and the tags are masked out of the loss):

```json
[
  {"messages": [
    {"role": "system",    "content": "You are a helpful assistant."},
    {"role": "user",      "content": "What does RMSNorm do?"},
    {"role": "assistant", "content": "It rescales a vector by its RMS ..."}
  ]}
]
```

See `data/sample_conversations.json` for a runnable example and `utils/chat.py`
for every accepted file layout.

---

## Supervised fine-tuning (SFT)

SFT is just **resuming a pretrained checkpoint on chat data**. Pretrain on text,
then fine-tune on conversations.

### With the CUDA trainer (recommended)

`--resume` reads the architecture back from the checkpoint, so the fine-tune run
only needs the new data / learning rate / steps:

```bash
cd cuda
# 1. pretrain on plain text
./build/train_transformer_cuda --corpus ../data/tiny_corpus.txt \
    --tokenizer bbpe --tokenizer-path ../tokenizer/tok_out/tokenizer.bbpe \
    --steps 2000

# 2. fine-tune on chat data — same tokenizer, lower lr, its own outputs
./build/train_transformer_cuda --resume checkpoints/latest.ckpt \
    --corpus ../data/sample_conversations.json --data-format chat \
    --tokenizer bbpe --tokenizer-path ../tokenizer/tok_out/tokenizer.bbpe \
    --lr 0.01 --steps 300 --reset-step \
    --checkpoint-dir sft_ckpts --metrics-path sft_metrics.csv
```

`--reset-step` restarts the step counter at 0 for the fine-tune run (weights are
kept). The tokenizer must match the one used for pretraining — the trainer
errors out early if the vocab sizes disagree.

### With the NumPy trainer

Pretrain, then resume with `--data-format chat` (re-supply the same
architecture flags):

```bash
python train.py --corpus data/tiny_corpus.txt --steps 2000 \
    --d-model 128 --num-heads 8 --num-layers 4 --d-ff 256 --seq-len 32
python train.py --resume checkpoints/latest.npz \
    --corpus data/sample_conversations.json --data-format chat \
    --d-model 128 --num-heads 8 --num-layers 4 --d-ff 256 --seq-len 32 \
    --lr 0.01 --steps 300
```

> Checkpoints are **not** interchangeable between the two trainers: `train.py`
> writes NumPy `.npz`, the CUDA trainer writes its own binary `.ckpt`. Pretrain
> and fine-tune with the same implementation.

---

## Tokenizer (optional — for `--tokenizer bbpe`)

If `tokenizer/tok_out/tokenizer.bbpe` is present (it may already be, but the
`tok_out/` directory is git-ignored, so a fresh clone won't have it — train one
with the steps at the end of this section if it's missing), the **CUDA trainer**
can use `--tokenizer bbpe` without any extra build (it compiles the tokenizer's
C++ core in). The NumPy trainer reaches the tokenizer through a Python
extension, which you build once:

```bash
cd tokenizer
bash build_py_client.sh          # builds & pip-installs the `bbpe_tokenizer` module
python ../test.py                # sanity-check encode/decode roundtrip
```

Then:

```bash
python train.py --tokenizer bbpe \
    --tokenizer-path tokenizer/tok_out/tokenizer.bbpe --vocab-size 32000 \
    --corpus data/tiny_corpus.txt --steps 200
```

To train a **new** tokenizer on your own corpus:

```bash
cd tokenizer
bash build.sh                    # builds the C++ CLI tools (needs CMake ≥ 3.16)
./build/train_tokenizer --data-dir ./your_corpus.txt \
    --output-dir ./tok_out --vocab-size 32000 --min-frequency 5
```

If `bbpe_tokenizer` isn't importable, `train.py` automatically falls back to the
byte tokenizer, so nothing above is required to get started.

---

## Common flags

Shared by both trainers unless noted:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--corpus PATH` | `data/tiny_corpus.txt` | training data file |
| `--data-format text\|chat` | `text` | plain text vs. conversation/SFT data |
| `--tokenizer byte\|bbpe` | `byte` | byte-level (no deps) or bbpe |
| `--tokenizer-path PATH` | `tokenizer/tok_out/tokenizer.bbpe` | bbpe model file |
| `--vocab-size N` | `32000` | output vocab (bbpe only) |
| `--d-model / --num-heads / --num-layers / --d-ff / --seq-len` | `512 / 8 / 24 / 2026 / 32` | model architecture |
| `--batch-size / --steps / --lr / --seed` | `8 / 500 / 0.05 / 0` | optimization (plain SGD) |
| `--log-every / --checkpoint-every` | `10 / 100` | logging & checkpoint cadence |
| `--checkpoint-dir / --metrics-path` | `checkpoints / metrics.csv` | output locations |
| `--resume PATH` | — | resume/fine-tune from a checkpoint |

CUDA-trainer-only: `--fp8` (enable simulated FP8), `--reset-step` (restart the
step counter on resume). On resume the CUDA trainer takes the architecture from
the checkpoint, so the arch flags above can be omitted.
