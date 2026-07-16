#!/usr/bin/env bash
# Resume training in the GPU training container (run ./setup-docker.sh once
# first). Reads/writes cuda/checkpoints/ on the host, so run ./setup.sh (or
# python download_models.py) beforehand to seed it from Hugging Face.
#
#   ./train-docker.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
docker compose up train
