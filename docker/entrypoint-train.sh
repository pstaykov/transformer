#!/usr/bin/env bash
# Builds the CUDA trainer on first run (cached afterwards in the cuda_build
# volume) and then execs it. Must run with GPU access (--gpus all /
# docker compose's device reservation) since CMAKE_CUDA_ARCHITECTURES=native
# needs a visible GPU to detect the compute capability at build time.
set -euo pipefail
cd /app/cuda

if [ ! -x build/train_transformer_cuda ]; then
    echo "[train] no cached build found, compiling (first run only)..."
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j"$(nproc)"
fi

exec ./build/train_transformer_cuda "$@"
