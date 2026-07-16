#!/usr/bin/env bash
# Resume training with the CUDA trainer, WITHOUT Docker. Needs an NVIDIA GPU +
# CUDA Toolkit + CMake on this machine (run ./setup.sh first to get the
# checkpoints/tokenizer). Builds cuda/build/train_transformer_cuda on first
# run (cached after that).
#
#   ./train.sh                      # resumes the base model on the tiny sample corpus
#   CORPUS=... ./train.sh           # point at a different corpus
#   ./train.sh --steps 5000 --lr 1e-4   # extra args are passed straight through
#
# To continue the real ~10GB corpus run instead of the tiny sample:
#   CORPUS=KEVINDATA/mein_trainingsdaten_10gb.txt DATA_FORMAT=text \
#   BATCH_SIZE=16 ./train.sh --lr 1e-4 --min-lr 1e-5 --warmup-steps 500 \
#       --grad-clip 1.0 --label-smoothing 0.05 --dropout 0.1
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-cuda/checkpoints/run3}"
RESUME="${RESUME:-$CHECKPOINT_DIR/latest.ckpt}"
CORPUS="${CORPUS:-data/tiny_corpus.txt}"
DATA_FORMAT="${DATA_FORMAT:-text}"
TOKENIZER_PATH="${TOKENIZER_PATH:-tokenizer/tok_out_kevindata/tokenizer.bbpe}"
METRICS_PATH="${METRICS_PATH:-$CHECKPOINT_DIR/metrics.csv}"
BATCH_SIZE="${BATCH_SIZE:-8}"

if [ ! -f "$RESUME" ]; then
    echo "No checkpoint at $RESUME - run ./setup.sh first (or set RESUME=path)." >&2
    exit 1
fi
if [ ! -f "$TOKENIZER_PATH" ]; then
    echo "No tokenizer at $TOKENIZER_PATH - run ./setup.sh first (or set TOKENIZER_PATH=path)." >&2
    exit 1
fi

echo "== Building the CUDA trainer (skipped if already built) =="
cd cuda
if [ ! -x build/train_transformer_cuda ]; then
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j"$(nproc)"
fi
cd ..

exec ./cuda/build/train_transformer_cuda \
    --resume "$RESUME" \
    --corpus "$CORPUS" \
    --data-format "$DATA_FORMAT" \
    --tokenizer bbpe --tokenizer-path "$TOKENIZER_PATH" \
    --batch-size "$BATCH_SIZE" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --metrics-path "$METRICS_PATH" \
    "$@"
