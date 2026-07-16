"""Precompute "how KEVIN talked at step N" samples for the showcase website.

Loading and forward-passing through eight ~83M-param checkpoints is too slow
to do live in serve.py on every page load, so this runs once, offline, and
writes a small JSON file the frontend fetches statically:

    python tools_gen_evolution.py --run-dir cuda/checkpoints/run3

Output: web/data/evolution.json, a list of {step, loss, perplexity, text}.
"""

import argparse
import json
import os

from model import Transformer
from utils.ckpt_convert import load_any
from utils.generate import generate
from utils.tokenizer import load_tokenizer

DEFAULT_STEPS = [50_000, 100_000, 200_000, 300_000, 400_000, 500_000, 600_000, 698_000]
DEFAULT_PROMPT = "The weather today is"


def _metrics_at_step(metrics_path, step):
    """Best-effort (loss, perplexity) lookup from metrics.csv for a given step."""
    if not os.path.exists(metrics_path):
        return None, None
    best = None
    with open(metrics_path, newline="") as f:
        import csv
        for row in csv.DictReader(f):
            try:
                row_step = int(row["step"])
            except (KeyError, ValueError):
                continue
            if row_step <= step:
                best = row
            else:
                break
    if best is None:
        return None, None
    return float(best["loss"]), float(best["perplexity"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", default="cuda/checkpoints/run3",
                    help="directory of ckpt_step<N>.ckpt files")
    p.add_argument("--steps", type=int, nargs="+", default=DEFAULT_STEPS)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--tokenizer-path", default="tokenizer/tok_out_kevindata/tokenizer.bbpe")
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-heads", type=int, default=8, help="only used for .npz checkpoints")
    p.add_argument("--metrics-path", default=None,
                    help="defaults to <run-dir>/metrics.csv")
    p.add_argument("-o", "--out", default="web/data/evolution.json")
    args = p.parse_args()

    metrics_path = args.metrics_path or os.path.join(args.run_dir, "metrics.csv")

    samples = []
    for step in args.steps:
        ckpt_path = os.path.join(args.run_dir, f"ckpt_step{step}.ckpt")
        if not os.path.exists(ckpt_path):
            print(f"[gen_evolution] skipping step {step}: no checkpoint at {ckpt_path}")
            continue

        print(f"[gen_evolution] loading {ckpt_path} ...")
        params, config, actual_step = load_any(ckpt_path, num_heads=args.num_heads)
        tokenizer, tok_vocab = load_tokenizer("bbpe", args.tokenizer_path, config["vocab_size"])
        if config["vocab_size"] != tok_vocab:
            raise ValueError(
                f"{ckpt_path}: checkpoint vocab_size={config['vocab_size']} but tokenizer "
                f"has vocab_size={tok_vocab}"
            )

        model = Transformer(
            vocab_size=config["vocab_size"],
            d_model=config["d_model"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            d_ff=config["d_ff"],
            max_len=config["max_len"],
        )
        model.load_params(params)

        # This is a base model completing raw text, not a chat turn - it has no
        # reason to stop early, so don't cut it off on chat role tags the way
        # serve.py's live chat endpoint does.
        text = generate(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p, seed=args.seed, stop_strings=(),
        )
        loss, perplexity = _metrics_at_step(metrics_path, actual_step)

        print(f"[gen_evolution] step {actual_step}: {args.prompt!r} -> {text!r}")
        samples.append({
            "step": actual_step,
            "loss": loss,
            "perplexity": perplexity,
            "text": text,
        })

    out = {"prompt": args.prompt, "samples": samples}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[gen_evolution] wrote {len(samples)} samples to {args.out}")


if __name__ == "__main__":
    main()
