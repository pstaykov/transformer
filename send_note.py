"""Send a one-off 'KEVIN update' email with a free-text note, e.g. from an
LLM or human watching a training run.

Doesn't need the model loaded - just reads metrics.csv (if present) for a
chart and the current step, and email.env for credentials.

Usage:
    python send_note.py "training running clean, loss trending down, GPU temps normal"
    python send_note.py "step 60000: perplexity dropped below 50" --step 60000
    python send_note.py --prompt "The" --completion "men in government need this ..." --step 60000
    # multiple samples: repeat --prompt/--completion in matching order
    python send_note.py --prompt "The" "Hello, I'm Kevin and" \\
                         --completion "men in government need this ..." "I like to talk about ..." \\
                         --step 60000
"""

import argparse

from utils import notify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("message", nargs="?", default=None,
                         help="free-text note to include in the email")
    parser.add_argument("--prompt", nargs="+", default=None,
                         help="one or more generation prompts, each paired positionally with --completion")
    parser.add_argument("--completion", nargs="+", default=None,
                         help="one or more generated continuations, each rendered in its own \"Kevin said\" box")
    parser.add_argument("--step", type=int, default=None,
                         help="step number to show (defaults to the last row in metrics.csv)")
    parser.add_argument("--metrics-path", default="metrics.csv")
    parser.add_argument("--email-config", default="email.env")
    args = parser.parse_args()

    if args.message is None and (args.prompt is None or args.completion is None):
        raise SystemExit("Provide either a free-text message or both --prompt and --completion.")
    if (args.prompt is None) != (args.completion is None):
        raise SystemExit("--prompt and --completion must be given together.")
    if args.prompt is not None and len(args.prompt) != len(args.completion):
        raise SystemExit("--prompt and --completion must have the same number of values.")

    config = notify.load_env_file(args.email_config)
    if not config:
        raise SystemExit(f"No email config found at {args.email_config}. See utils/notify.py.")

    samples = list(zip(args.prompt, args.completion)) if args.prompt else None
    notify.send_note(args.message or "", config, step=args.step, metrics_path=args.metrics_path,
                      samples=samples)
    print("sent.")


if __name__ == "__main__":
    main()
