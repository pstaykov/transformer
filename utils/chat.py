import json

"""Conversation-format data loading for supervised fine-tuning.

Accepts JSON/JSONL where each conversation is a list of
{"role": "system"|"user"|"assistant", "content": "..."} messages - the same
shape used by the OpenAI chat API and most SFT datasets. Several common file
layouts are auto-detected (see load_conversations).

Each conversation is rendered into a flat token stream with simple role tags:

    <|system|>
    {content}
    <|user|>
    {content}
    <|assistant|>
    {content}

Only tokens that fall inside an assistant turn are valid next-token
prediction targets (train.py masks the loss elsewhere with IGNORE_INDEX) -
the model should learn to produce assistant replies, not to reproduce the
user's side of the conversation or the role tags themselves.
"""

IGNORE_INDEX = -100

ROLE_TAGS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
}

# One of tokenizer/tok_out_kevindata's 5 reserved special ids (32000-32004,
# see tokenizer/tools/remap_specials.cpp) - encodes as a single token id
# rather than ordinary BPE text. Appended after every assistant turn so the
# model has an explicit, single-token way to signal "turn over" instead of
# relying on it correctly spelling out the next role tag as literal text
# (see utils/generate.py's DEFAULT_STOP_STRINGS for the matching stop logic).
EOS_TAG = "<|endoftext|>"


def load_conversations(path):
    """Load conversations from a .json or .jsonl file.

    Supported shapes:
      - JSON array of {"messages": [...]} objects
      - JSON array of conversations, each itself a list of message dicts
      - JSON array of message dicts (a single conversation)
      - JSONL: one of the above per line (each line is one conversation,
        either {"messages": [...]} or a bare list of message dicts)

    Returns:
        list of conversations, each a list of {"role", "content"} dicts.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if path.endswith(".jsonl"):
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        records = json.loads(text)
        if isinstance(records, dict):
            records = [records]

    # A bare list of message dicts is a single conversation.
    if records and isinstance(records[0], dict) and "role" in records[0]:
        return [records]

    conversations = []
    for rec in records:
        messages = rec["messages"] if isinstance(rec, dict) else rec
        conversations.append(messages)
    return conversations


def render_conversation(tokenizer, messages):
    """Tokenize one conversation.

    Returns:
        (token_ids: list[int], predict_mask: list[bool]) of equal length -
        predict_mask[i] is True where token_ids[i] is part of an assistant
        turn's content (i.e. a valid next-token prediction target).
    """
    ids, mask = [], []
    for m in messages:
        role, content = m["role"], m["content"]
        tag = ROLE_TAGS.get(role, f"<|{role}|>")

        tag_ids = tokenizer.encode(f"{tag}\n")
        ids.extend(tag_ids)
        mask.extend([False] * len(tag_ids))

        content_ids = tokenizer.encode(f"{content}\n")
        ids.extend(content_ids)
        mask.extend([role == "assistant"] * len(content_ids))

        if role == "assistant":
            eos_ids = tokenizer.encode(EOS_TAG)
            ids.extend(eos_ids)
            mask.extend([True] * len(eos_ids))

    return ids, mask


def build_dataset(tokenizer, conversations):
    """Render and concatenate every conversation into one token stream.

    Returns:
        (ids: list[int], predict_mask: list[bool])
    """
    all_ids, all_mask = [], []
    for messages in conversations:
        ids, mask = render_conversation(tokenizer, messages)
        all_ids.extend(ids)
        all_mask.extend(mask)
    return all_ids, all_mask
