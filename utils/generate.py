"""Autoregressive sampling for the from-scratch transformer.

The trainer never needed this - utils/notify.py has a greedy argmax loop for its
progress emails, and that's all the inference code that existed. This adds
temperature / top-k / top-p sampling, a context window that respects the model's
learned positional embeddings, stop-string handling, and token-at-a-time
streaming so a server can forward tokens as they're produced.

Two properties of the model shape this code:

  * There is no KV cache. Every new token re-runs a full forward pass over the
    whole context, so generation cost is O(context) per token and streaming is
    the natural interface - each token genuinely arrives one at a time.

  * There is no EOS token. utils/chat.py's role tags ("<|assistant|>" etc.) are
    byte/BPE-encoded as ordinary text rather than registered as special tokens,
    so a model that learns the chat format signals "my turn is over" by emitting
    the literal text "<|user|>". Stopping is therefore a string match on the
    decoded output, not a token id check.
"""

import numpy as np

from utils.chat import ROLE_TAGS

# A well-trained chat model ends its turn by starting the next role tag.
DEFAULT_STOP_STRINGS = (ROLE_TAGS["user"], ROLE_TAGS["system"], ROLE_TAGS["assistant"])


def render_chat_prompt(messages, system=None):
    """Build the prompt text for a chat completion.

    Mirrors utils/chat.py::render_conversation exactly - the model only ever saw
    this layout during SFT - and leaves the trailing "<|assistant|>\\n" open for
    the model to complete.
    """
    parts = []
    if system:
        parts.append(f"{ROLE_TAGS['system']}\n{system}\n")
    for m in messages:
        tag = ROLE_TAGS.get(m["role"], f"<|{m['role']}|>")
        parts.append(f"{tag}\n{m['content']}\n")
    parts.append(f"{ROLE_TAGS['assistant']}\n")
    return "".join(parts)


def render_chat_prompt_continue(messages, system=None):
    """Build the prompt for the "Continue" button: keep generating the last
    message (an assistant reply that already stopped) instead of opening a
    fresh turn. Same layout as render_chat_prompt for every message before
    the last one, but the last message's content has no trailing newline -
    generation picks up exactly where it left off, mid-turn, rather than
    starting a new <|assistant|> block.
    """
    parts = []
    if system:
        parts.append(f"{ROLE_TAGS['system']}\n{system}\n")
    for m in messages[:-1]:
        tag = ROLE_TAGS.get(m["role"], f"<|{m['role']}|>")
        parts.append(f"{tag}\n{m['content']}\n")
    last = messages[-1]
    tag = ROLE_TAGS.get(last["role"], f"<|{last['role']}|>")
    parts.append(f"{tag}\n{last['content']}")
    return "".join(parts)


def sample_from_logits(logits, temperature, top_k, top_p, rng):
    """Pick one token id from a (vocab_size,) logit vector."""
    if temperature <= 0:
        return int(np.argmax(logits))

    logits = logits.astype(np.float64) / temperature

    if top_k:
        k = min(int(top_k), logits.shape[-1])
        kept = np.argpartition(logits, -k)[-k:]
        masked = np.full_like(logits, -np.inf)
        masked[kept] = logits[kept]
        logits = masked

    probs = np.exp(logits - logits.max())
    probs /= probs.sum()

    if top_p and 0 < top_p < 1.0:
        order = np.argsort(probs)[::-1]
        cumulative = np.cumsum(probs[order])
        # Keep the smallest prefix whose mass exceeds top_p (always >= 1 token).
        cutoff = int(np.searchsorted(cumulative, top_p)) + 1
        keep = order[:cutoff]
        masked = np.zeros_like(probs)
        masked[keep] = probs[keep]
        probs = masked / masked.sum()

    return int(rng.choice(len(probs), p=probs))


def _held_back(text, stop_strings):
    """How many trailing characters aren't safe to emit yet.

    Two things must be withheld from a streamed chunk:

      * A suffix that is a proper prefix of a stop string. "<|user" has to be
        held until we know whether the next token completes "<|user|>" (stop) or
        turns it into something else (emit). Without this the stop tag leaks out
        one character at a time before the match ever fires.

      * A trailing U+FFFD. The byte tokenizer emits one token per byte, so a
        multi-byte codepoint decodes to a replacement character until its last
        byte arrives - emitting it would put a permanent mojibake in the output.
    """
    hold = 0
    while hold < len(text) and text[len(text) - 1 - hold] == "�":
        hold += 1

    for stop in stop_strings:
        for k in range(1, min(len(stop), len(text))):
            if text.endswith(stop[:k]):
                hold = max(hold, k)
    return hold


def generate_stream(model, tokenizer, prompt, max_new_tokens=64, temperature=0.8,
                    top_k=40, top_p=0.95, max_len=None, stop_strings=DEFAULT_STOP_STRINGS,
                    seed=None):
    """Yield the completion incrementally, one decoded text delta per token.

    max_len is the model's context window. It is a hard limit, not a suggestion:
    Embedding.pos_emb has shape (max_len, d_model) and slicing past it silently
    produces a mismatched positional slice, so the context is always truncated
    to the most recent max_len tokens.
    """
    if max_len is None:
        max_len = model.embedding.pos_emb.shape[0]

    # If a stop string is registered as a single special token (see
    # tokenizer/tools/remap_specials.cpp - tok_out_kevindata's tokenizer.bbpe
    # repurposes 3 reserved slots for the real role tags), decoding it with
    # skip_special_tokens=True (below) yields "" rather than the tag text, so
    # the string search a few lines down would never see it. Stop on the
    # sampled id directly for any stop string that tokenizes to exactly one
    # id; the string search stays as the only mechanism for stop strings that
    # aren't (yet) registered as specials, e.g. the byte tokenizer never has
    # any.
    stop_token_ids = set()
    for s in stop_strings:
        enc = list(tokenizer.encode(s))
        if len(enc) == 1:
            stop_token_ids.add(enc[0])

    rng = np.random.default_rng(seed)
    ids = list(tokenizer.encode(prompt))
    generated = []
    emitted = 0  # characters of the completion already yielded

    for _ in range(max_new_tokens):
        context = ids[-max_len:]
        logits = model.forward(np.array([context], dtype=np.int64))
        next_id = sample_from_logits(logits[0, -1], temperature, top_k, top_p, rng)

        if next_id in stop_token_ids:
            decoded = tokenizer.decode(generated)
            if len(decoded) > emitted:
                yield decoded[emitted:]
            return

        ids.append(next_id)
        generated.append(next_id)

        # Decode the whole completion each step and emit the new suffix. Decoding
        # tokens one at a time wouldn't work: a token can be half of a UTF-8
        # codepoint, and a stop tag spans several tokens.
        decoded = tokenizer.decode(generated)

        stop_at = min((decoded.find(s) for s in stop_strings if s in decoded), default=-1)
        if stop_at != -1:
            if stop_at > emitted:
                yield decoded[emitted:stop_at]
            return

        safe = len(decoded) - _held_back(decoded, stop_strings)
        if safe > emitted:
            yield decoded[emitted:safe]
            emitted = safe

    # Budget exhausted: flush whatever was held back but never completed a stop tag.
    decoded = tokenizer.decode(generated)
    if len(decoded) > emitted:
        yield decoded[emitted:]


def generate(model, tokenizer, prompt, **kwargs):
    """Non-streaming wrapper: returns the full completion as one string."""
    return "".join(generate_stream(model, tokenizer, prompt, **kwargs))
