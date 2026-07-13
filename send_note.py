"""Send a one-off 'KEVIN update' email with a free-text note, e.g. from an
LLM or human watching a training run.

Doesn't need the model loaded - just reads metrics.csv (if present) for a
chart and the current step, and email.env for credentials.

Usage:
    python send_note.py "training running clean, loss trending down, GPU temps normal"
    python send_note.py "step 60000: perplexity dropped below 50" --step 60000
"""

import argparse

from utils import notify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("message", help="free-text note to include in the email")
    parser.add_argument("--step", type=int, default=None,
                         help="step number to show (defaults to the last row in metrics.csv)")
    parser.add_argument("--metrics-path", default="metrics.csv")
    parser.add_argument("--email-config", default="email.env")
    args = parser.parse_args()

    config = notify.load_env_file(args.email_config)
    if not config:
        raise SystemExit(f"No email config found at {args.email_config}. See utils/notify.py.")

    notify.send_note(args.message, config, step=args.step, metrics_path=args.metrics_path)
    print("sent.")


if __name__ == "__main__":
    main()
