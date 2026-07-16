# Training container: Ubuntu + CUDA toolkit + cmake, builds and runs
# cuda/train_transformer_cuda. Needs a GPU at runtime (--gpus all).
FROM nvidia/cuda:12.6.2-devel-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        build-essential \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Only the sources needed to build - data/checkpoints/tokenizer-output are
# mounted as volumes at runtime (see docker-compose.yml), so the image stays
# small and doesn't need rebuilding when a training run produces new data.
COPY cuda/CMakeLists.txt ./cuda/CMakeLists.txt
COPY cuda/src ./cuda/src
COPY cuda/include ./cuda/include
COPY tokenizer/src ./tokenizer/src
COPY tokenizer/include ./tokenizer/include
COPY data ./data
COPY docker/entrypoint-train.sh ./docker/entrypoint-train.sh
RUN chmod +x ./docker/entrypoint-train.sh

ENTRYPOINT ["./docker/entrypoint-train.sh"]
# No args: builds and runs the bundled tiny_corpus sample with default flags,
# just to prove the GPU + build work. Override the command for a real run,
# e.g. to resume training on KEVINDATA (see README's Docker section).
CMD []
