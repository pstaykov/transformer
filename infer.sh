#!/usr/bin/env bash
# Start the showcase site + chat server, WITHOUT Docker (runs serve.py on
# CPU with the NumPy model). Run ./setup.sh first to get the checkpoints/
# tokenizer. Open http://localhost:8000 (or your LAN IP) once it's up.
#
#   ./infer.sh
#   ./infer.sh --host 127.0.0.1     # localhost only instead of the whole LAN
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ -d .venv ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

CKPT="${CKPT:-cuda/checkpoints/run3/latest.ckpt}"
CHAT_CKPT="${CHAT_CKPT:-cuda/checkpoints/sft_wildchat/latest.ckpt}"
TOKENIZER_PATH="${TOKENIZER_PATH:-tokenizer/tok_out_kevindata/tokenizer.bbpe}"
VOCAB_SIZE="${VOCAB_SIZE:-32005}"
METRICS_PATH="${METRICS_PATH:-cuda/checkpoints/run3/metrics.csv}"

if [ ! -f "$CKPT" ]; then
    echo "No checkpoint at $CKPT - run ./setup.sh first (or set CKPT=path)." >&2
    exit 1
fi

exec python serve.py \
    --ckpt "$CKPT" \
    --tokenizer bbpe --tokenizer-path "$TOKENIZER_PATH" \
    --vocab-size "$VOCAB_SIZE" \
    --metrics-path "$METRICS_PATH" \
    --chat-ckpt "$CHAT_CKPT" \
    --host 0.0.0.0 --port 8000 \
    "$@"
