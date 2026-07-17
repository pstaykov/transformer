#!/usr/bin/env bash
# One-shot setup for running WITHOUT Docker: creates a venv, installs Python
# deps, builds the bbpe tokenizer's Python extension, and downloads the
# pretrained checkpoints + tokenizer from Hugging Face (kevinindustries/kevin-k2,
# kevinindustries/kevin-chat) into cuda/checkpoints/ and tokenizer/tok_out_kevindata/.
#
#   ./setup.sh
#
# After this: ./train.sh to resume training, ./infer.sh to start the showcase
# / chat server. For Docker instead, see ./setup-docker.sh.
#
# Any extra arguments are passed through to tools/download_models.py, e.g.:
#   ./setup.sh --skip-chat        # only fetch the base model
#   ./setup.sh --force            # re-download even if files already exist
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "== Python virtualenv (.venv) =="
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "== Installing Python dependencies =="
pip install --upgrade pip >/dev/null
pip install -r requirements.txt
pip install huggingface_hub

echo "== Building the bbpe tokenizer's Python extension =="
pip install pybind11 setuptools >/dev/null
if pip install ./tokenizer --no-build-isolation; then
    echo "tokenizer extension installed."
else
    echo "tokenizer extension failed to build - training/inference will fall back to the byte tokenizer." >&2
fi

echo "== Downloading pretrained checkpoints + tokenizer from Hugging Face =="
python tools/download_models.py "$@"

cat <<'EOF'

Done. Quick reference:
  Train (no Docker):      ./train.sh
  Inference (no Docker):  ./infer.sh
  Train (Docker, GPU):    ./setup-docker.sh   (once)   then  ./train-docker.sh
  Inference (Docker):     ./setup-docker.sh   (once)   then  ./infer-docker.sh
EOF
