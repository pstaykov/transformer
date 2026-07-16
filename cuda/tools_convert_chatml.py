#!/usr/bin/env python3
"""Convert the <im_start>role / <im_end> ChatML-style transcript on the
external SSD into the JSONL conversation format expected by
train_transformer_cuda --data-format chat (one {"messages": [...]} object
per line, matching utils/chat.py::load_conversations)."""
import json
import sys

def main(src, dst):
    conversations = []
    cur = None
    role = None
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("<im_start>"):
                role = line[len("<im_start>"):].strip()
                if role == "system":
                    if cur:
                        conversations.append(cur)
                    cur = []
                continue
            if cur is None:
                continue
            content = line
            if content.endswith("<im_end>"):
                content = content[: -len("<im_end>")]
            content = content.strip()
            cur.append({"role": role, "content": content})

    if cur:
        conversations.append(cur)

    with open(dst, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps({"messages": conv}) + "\n")

    print(f"wrote {len(conversations)} conversations to {dst}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
