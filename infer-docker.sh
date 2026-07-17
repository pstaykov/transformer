#!/usr/bin/env bash
# Start the showcase site + chat server in the inference container (run
# ./setup-docker.sh once first). Reads cuda/checkpoints/ and
# tokenizer/tok_out_kevindata/ from the host - run ./setup.sh (or
# python tools/download_models.py) beforehand to seed them from Hugging Face.
#
#   ./infer-docker.sh
# Then open http://localhost:8000
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
docker compose up inference
