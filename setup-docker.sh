#!/usr/bin/env bash
# One-shot setup for the training + inference containers.
#
#   ./setup-docker.sh
#
# Installs the NVIDIA Container Toolkit if it's missing (needed for the
# training container's --gpus access), then builds both images.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is not installed. Install it first: https://docs.docker.com/engine/install/ubuntu/" >&2
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    echo "Docker daemon isn't reachable (permissions? not running?). Try: sudo systemctl start docker" >&2
    exit 1
fi

echo "== Checking for the NVIDIA Container Toolkit (GPU access inside containers) =="
if docker info 2>/dev/null | grep -qi nvidia; then
    echo "already configured, skipping."
elif command -v nvidia-ctk >/dev/null 2>&1; then
    echo "nvidia-ctk found, wiring up the docker runtime..."
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
else
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "No NVIDIA driver detected (nvidia-smi missing)." >&2
        echo "The training container needs a GPU; install the driver first, then re-run this script." >&2
        exit 1
    fi
    echo "installing nvidia-container-toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
        | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
fi

echo
echo "== Building images =="
docker compose build

echo
echo "== Fetching pretrained checkpoints + tokenizer from Hugging Face =="
if command -v python3 >/dev/null 2>&1; then
    python3 -m pip show huggingface_hub >/dev/null 2>&1 || python3 -m pip install --user huggingface_hub
    python3 tools/download_models.py
else
    echo "python3 not found on the host - skipping. Run 'python tools/download_models.py' manually" >&2
    echo "(it just needs huggingface_hub; it doesn't need the rest of requirements.txt)." >&2
fi

cat <<'EOF'

Done. Quick reference:

  Training (GPU, resumable):    ./train-docker.sh
  Inference (showcase + chat):  ./infer-docker.sh   then open http://localhost:8000

Both containers read/write the same cuda/checkpoints/, tokenizer/tok_out*/,
KEVINDATA/, and data/ folders on the host, so nothing needs to be copied in or
out. Re-run 'python3 tools/download_models.py --force' any time to refresh the
checkpoints/tokenizer from Hugging Face.
EOF
