#!/usr/bin/env python3
"""Download allenai/WildChat-1M, keep only English-language conversations,
and write them out as JSONL {"messages": [...]} conversations - the same
format train_transformer_cuda --data-format chat expects (matches
utils/chat.py::load_conversations)."""
import json
import sys

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO_ID = "allenai/WildChat-1M"
NUM_SHARDS = 14


def main(dst):
    total_kept = 0
    total_seen = 0
    with open(dst, "w", encoding="utf-8") as out:
        for i in range(NUM_SHARDS):
            fname = f"data/train-{i:05d}-of-{NUM_SHARDS:05d}.parquet"
            path = hf_hub_download(REPO_ID, fname, repo_type="dataset")
            table = pq.read_table(path, columns=["conversation", "language"])
            convs = table.column("conversation").to_pylist()
            langs = table.column("language").to_pylist()
            total_seen += len(convs)

            for conv, lang in zip(convs, langs):
                if lang != "English":
                    continue
                messages = [{"role": turn["role"], "content": turn["content"]} for turn in conv]
                out.write(json.dumps({"messages": messages}) + "\n")
                total_kept += 1

            print(f"shard {i+1}/{NUM_SHARDS}: kept {total_kept}/{total_seen} so far", flush=True)

    print(f"wrote {total_kept} English conversations (of {total_seen} total) to {dst}")


if __name__ == "__main__":
    main(sys.argv[1])
