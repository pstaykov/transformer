#!/usr/bin/env python3
"""Download the pretrained checkpoints + tokenizer from Hugging Face and place
them where train.py, the CUDA trainer, and serve.py all expect to find them:

    kevinindustries/kevin-k2   (base model)  -> cuda/checkpoints/run3/latest.ckpt
                                              -> tokenizer/tok_out_kevindata/*
    kevinindustries/kevin-chat (SFT chat)    -> cuda/checkpoints/sft_wildchat/latest.ckpt

Run directly (python download_models.py) or via ./setup.sh, which also sets
up the venv and Python deps first.
"""
import argparse
import os
import shutil

from huggingface_hub import hf_hub_download

HERE = os.path.dirname(os.path.abspath(__file__))

TOKENIZER_DIR = os.path.join(HERE, "tokenizer", "tok_out_kevindata")
BASE_CKPT_DIR = os.path.join(HERE, "cuda", "checkpoints", "run3")
CHAT_CKPT_DIR = os.path.join(HERE, "cuda", "checkpoints", "sft_wildchat")

TOKENIZER_FILES = ["tokenizer/tokenizer.bbpe", "tokenizer/vocab.json", "tokenizer/merges.txt"]


def fetch(repo_id, filename, dest_dir, force=False):
    dest_path = os.path.join(dest_dir, os.path.basename(filename))
    if os.path.exists(dest_path) and not force:
        print(f"  already have {dest_path} (use --force to re-download)")
        return
    os.makedirs(dest_dir, exist_ok=True)
    local_path = hf_hub_download(repo_id=repo_id, filename=filename)
    shutil.copyfile(local_path, dest_path)
    print(f"  {repo_id}:{filename} -> {dest_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-repo", default="kevinindustries/kevin-k2",
                         help="base pretrained model + tokenizer (default: %(default)s)")
    parser.add_argument("--chat-repo", default="kevinindustries/kevin-chat",
                         help="SFT chat checkpoint (default: %(default)s)")
    parser.add_argument("--skip-base", action="store_true", help="don't download the base model/tokenizer")
    parser.add_argument("--skip-chat", action="store_true", help="don't download the chat checkpoint")
    parser.add_argument("--force", action="store_true", help="re-download even if the file already exists")
    args = parser.parse_args()

    if not args.skip_base:
        print(f"== base model + tokenizer ({args.base_repo}) ==")
        fetch(args.base_repo, "latest.ckpt", BASE_CKPT_DIR, force=args.force)
        for f in TOKENIZER_FILES:
            fetch(args.base_repo, f, TOKENIZER_DIR, force=args.force)

    if not args.skip_chat:
        print(f"== chat (SFT) checkpoint ({args.chat_repo}) ==")
        fetch(args.chat_repo, "latest.ckpt", CHAT_CKPT_DIR, force=args.force)

    print("\nDone.")
    print(f"  base checkpoint : {os.path.join(BASE_CKPT_DIR, 'latest.ckpt')}")
    print(f"  chat checkpoint : {os.path.join(CHAT_CKPT_DIR, 'latest.ckpt')}")
    print(f"  tokenizer       : {TOKENIZER_DIR}/")


if __name__ == "__main__":
    main()
