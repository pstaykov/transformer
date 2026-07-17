#!/usr/bin/env python3
"""Download HuggingFaceTB/everyday-conversations-llama3.1-2k (both splits) and
write it out as JSONL {"messages": [...]} conversations - the format
train_transformer_cuda --data-format chat expects (matches
utils/chat.py::load_conversations).

Usage: python tools/convert_everyday_conversations.py data/everyday_conversations.jsonl
(this is how data/everyday_conversations.jsonl was originally generated - it's
already in the repo, so you only need to re-run this if you want to refresh it)."""
import json
import sys

from datasets import load_dataset

REPO_ID = "HuggingFaceTB/everyday-conversations-llama3.1-2k"


def main(dst):
    ds = load_dataset(REPO_ID)
    total = 0
    with open(dst, "w", encoding="utf-8") as out:
        for split in ds:
            for row in ds[split]:
                messages = [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
                out.write(json.dumps({"messages": messages}) + "\n")
                total += 1
    print(f"wrote {total} conversations to {dst}")


if __name__ == "__main__":
    main(sys.argv[1])
