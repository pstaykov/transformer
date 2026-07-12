"""Train the from-scratch decoder-only transformer on a text corpus, or on
conversation-format SFT data (--data-format chat): JSON/JSONL of messages
with "role" (system/user/assistant) and "content" fields, the same shape
used by the OpenAI chat API. See utils/chat.py for supported file layouts;
data/sample_conversations.json is a runnable example.

Uses the bbpe_tokenizer bindings when importable, otherwise falls back to a
plain byte-level tokenizer so the script runs with nothing but numpy.

Logs step/loss/perplexity to both the console and a metrics.csv file, and
periodically writes model checkpoints (.npz) that can be resumed with
--resume.
"""

import argparse
import csv
import os
import time

import numpy as np

from model import Transformer
from utils.losses import CrossEntropyLoss
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils import fp8
from utils import chat as chat_utils

try:
    import bbpe_tokenizer
    HAS_BBPE = True
except ImportError:
    HAS_BBPE = False


class ByteTokenizer:
    """Fallback tokenizer: every byte of UTF-8 text is its own token."""

    vocab_size = 257  # 256 byte values + 1 pad id
    pad_id = 256

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(b for b in ids if b < 256).decode("utf-8", errors="replace")


def get_tokenizer(args):
    if args.tokenizer == "bbpe":
        if not HAS_BBPE:
            raise RuntimeError(
                "bbpe_tokenizer is not importable in this environment; "
                "run with the tokenizer/venv interpreter, or pass --tokenizer byte."
            )
        tok = bbpe_tokenizer.Tokenizer.load_binary(args.tokenizer_path)
        return tok, args.vocab_size
    tok = ByteTokenizer()
    return tok, tok.vocab_size


def load_corpus_ids(tokenizer, corpus_path):
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()
    ids = np.array(tokenizer.encode(text), dtype=np.int64)
    mask = np.ones_like(ids, dtype=bool)  # every token is a valid target
    return ids, mask


def load_chat_ids(tokenizer, corpus_path):
    conversations = chat_utils.load_conversations(corpus_path)
    ids, mask = chat_utils.build_dataset(tokenizer, conversations)
    print(f"Loaded {len(conversations)} conversations, "
          f"{sum(mask)}/{len(mask)} tokens are assistant-turn prediction targets")
    return np.array(ids, dtype=np.int64), np.array(mask, dtype=bool)


def sample_batch(ids, mask, batch_size, seq_len, rng):
    """Random contiguous windows for next-token prediction.

    Targets at positions where mask[t+1] is False (e.g. user/system turns
    in chat data) are replaced with IGNORE_INDEX so the loss skips them.
    """
    max_start = len(ids) - seq_len - 1
    starts = rng.integers(0, max_start, size=batch_size)
    x = np.stack([ids[s:s + seq_len] for s in starts])
    y = np.stack([ids[s + 1:s + seq_len + 1] for s in starts])
    y_mask = np.stack([mask[s + 1:s + seq_len + 1] for s in starts])
    y = np.where(y_mask, y, chat_utils.IGNORE_INDEX)
    return x, y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/tiny_corpus.txt")
    parser.add_argument("--data-format", choices=["text", "chat"], default="text",
                         help="'chat' expects JSON/JSONL conversations with role/content fields")
    parser.add_argument("--tokenizer", choices=["bbpe", "byte"], default="byte")
    parser.add_argument("--tokenizer-path", default="tokenizer/tok_out/tokenizer.bbpe")
    parser.add_argument("--vocab-size", type=int, default=32000, help="only used with --tokenizer bbpe")

    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=24)
    parser.add_argument("--d-ff", type=int, default=2026)
    parser.add_argument("--seq-len", type=int, default=32)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--metrics-path", default="metrics.csv")
    parser.add_argument("--resume", default=None, help="path to a .npz checkpoint to resume from")

    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    print("Warming up numba JIT for FP8 quantization...")
    t_warmup = time.time()
    fp8.warmup()
    print(f"  done in {time.time() - t_warmup:.2f}s")

    tokenizer, vocab_size = get_tokenizer(args)
    if args.data_format == "chat":
        ids, mask = load_chat_ids(tokenizer, args.corpus)
    else:
        ids, mask = load_corpus_ids(tokenizer, args.corpus)
    if len(ids) <= args.seq_len + 1:
        raise ValueError(
            f"corpus has only {len(ids)} tokens, need > seq_len+1 ({args.seq_len + 1})"
        )
    print(f"Loaded corpus: {len(ids)} tokens, vocab_size={vocab_size}")

    model = Transformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        max_len=args.seq_len,
    )
    loss_fn = CrossEntropyLoss(ignore_index=chat_utils.IGNORE_INDEX)

    start_step = 0
    if args.resume:
        start_step, meta = load_checkpoint(args.resume, model)
        print(f"Resumed from {args.resume} at step {start_step} (meta={meta})")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    metrics_is_new = not os.path.exists(args.metrics_path)
    metrics_file = open(args.metrics_path, "a", newline="")
    metrics_writer = csv.writer(metrics_file)
    if metrics_is_new:
        metrics_writer.writerow(["step", "loss", "perplexity", "lr", "tokens_per_sec", "elapsed_sec"])

    print(f"Training for {args.steps} steps (starting at {start_step})...")
    t_start = time.time()

    for step in range(start_step, start_step + args.steps):
        t0 = time.time()

        x, y = sample_batch(ids, mask, args.batch_size, args.seq_len, rng)
        logits = model.forward(x)
        loss = loss_fn.forward(logits, y)
        dlogits = loss_fn.backward()
        model.backward(dlogits)
        model.update(args.lr)

        dt = time.time() - t0
        tokens_per_sec = (args.batch_size * args.seq_len) / max(dt, 1e-9)
        perplexity = float(np.exp(min(loss, 20)))
        elapsed = time.time() - t_start

        metrics_writer.writerow([step, f"{loss:.6f}", f"{perplexity:.4f}", args.lr,
                                  f"{tokens_per_sec:.1f}", f"{elapsed:.2f}"])
        metrics_file.flush()

        if step % args.log_every == 0 or step == start_step + args.steps - 1:
            print(f"step {step:6d} | loss {loss:.4f} | ppl {perplexity:8.2f} "
                  f"| {tokens_per_sec:8.1f} tok/s | elapsed {elapsed:6.1f}s")

        if step > start_step and step % args.checkpoint_every == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"ckpt_step{step}.npz")
            save_checkpoint(ckpt_path, model, step, extra={"loss": loss})
            save_checkpoint(os.path.join(args.checkpoint_dir, "latest.npz"), model, step, extra={"loss": loss})
            print(f"  saved checkpoint -> {ckpt_path}")

    final_step = start_step + args.steps - 1
    final_ckpt = os.path.join(args.checkpoint_dir, f"ckpt_step{final_step}.npz")
    save_checkpoint(final_ckpt, model, final_step, extra={"loss": loss})
    save_checkpoint(os.path.join(args.checkpoint_dir, "latest.npz"), model, final_step, extra={"loss": loss})
    print(f"Final checkpoint -> {final_ckpt}")

    metrics_file.close()


if __name__ == "__main__":
    main()
