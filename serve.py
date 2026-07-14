"""Local showcase server for the from-scratch transformer.

Serves a single static page (web/) plus a small JSON API over the NumPy model:

    GET  /              the showcase page
    GET  /api/model     architecture, parameter count, training step, load status
    GET  /api/metrics   the training curves parsed out of metrics.csv
    POST /api/chat      a chat completion, streamed token-by-token over SSE

Checkpoints from either trainer work: pass a CUDA .ckpt (which carries its own
architecture in the header) or a train.py .npz (which doesn't - see --num-heads).

The server starts even with no checkpoint on disk. The showcase half of the page
is static content and stays useful; the chat half reports that no weights are
loaded rather than serving noise from a randomly-initialized model.

    python serve.py --ckpt checkpoints/latest.ckpt
"""

import argparse
import csv
import json
import os
import time

import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from utils import fp8

# FP8 is a training-numerics feature: it simulates E4M3 by snapping every matmul
# operand onto the FP8 grid, which in pure numpy costs more than it saves at
# batch size 1. Inference runs in plain float32. This must be set before the
# model is built, and before utils.fp8's numba kernel is ever called.
fp8.ENABLED = False

from model import Transformer                       # noqa: E402
from utils.checkpoint import count_params           # noqa: E402
from utils.ckpt_convert import load_any             # noqa: E402
from utils.generate import generate_stream, render_chat_prompt  # noqa: E402
from utils.tokenizer import load_tokenizer          # noqa: E402

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

app = FastAPI(title="KEVIN")

# Populated by load_model() at startup; the API reports model_loaded=False until
# then, and the chat endpoint refuses to generate.
STATE = {
    "model": None,
    "tokenizer": None,
    "config": None,
    "step": 0,
    "params": 0,
    "ckpt_path": None,
    "tokenizer_name": None,
    "error": None,
    "tokens_per_sec": None,
    "metrics_path": "metrics.csv",
}


class ChatRequest(BaseModel):
    messages: list[dict]
    max_new_tokens: int = 96
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    system: str | None = None
    seed: int | None = None


def load_model(ckpt_path, tokenizer_kind, tokenizer_path, num_heads, vocab_size):
    """Load the tokenizer and (if it exists) the checkpoint into STATE."""
    tokenizer, tok_vocab = load_tokenizer(tokenizer_kind, tokenizer_path, vocab_size)
    STATE["tokenizer"] = tokenizer
    STATE["tokenizer_name"] = (
        f"bbpe ({os.path.basename(tokenizer_path)})" if tokenizer_kind == "bbpe"
        else "byte-level fallback"
    )
    STATE["ckpt_path"] = ckpt_path

    if not os.path.exists(ckpt_path):
        STATE["error"] = f"no checkpoint at {ckpt_path}"
        print(f"[serve] {STATE['error']} - showcase will load, chat is disabled")
        return

    print(f"[serve] loading {ckpt_path} ...")
    t0 = time.time()
    params, config, step = load_any(ckpt_path, num_heads=num_heads)

    if config["vocab_size"] != tok_vocab:
        STATE["error"] = (
            f"checkpoint vocab_size={config['vocab_size']} but the "
            f"{STATE['tokenizer_name']} tokenizer has vocab_size={tok_vocab}. "
            "The tokenizer must match the one used for training."
        )
        print(f"[serve] {STATE['error']}")
        return

    model = Transformer(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        d_ff=config["d_ff"],
        max_len=config["max_len"],
    )
    model.load_params(params)

    STATE["model"] = model
    STATE["config"] = config
    STATE["step"] = step
    STATE["params"] = count_params(model.params())
    STATE["error"] = None

    print(f"[serve] loaded {STATE['params']:,} params (step {step}) "
          f"in {time.time() - t0:.1f}s: {config}")


@app.get("/api/model")
def api_model():
    cfg = STATE["config"] or {}
    return {
        "name": "KEVIN",
        "model_loaded": STATE["model"] is not None,
        "error": STATE["error"],
        "ckpt_path": STATE["ckpt_path"],
        "tokenizer": STATE["tokenizer_name"],
        "step": STATE["step"],
        "params": STATE["params"],
        "tokens_per_sec": STATE["tokens_per_sec"],
        "config": cfg,
        "d_head": cfg["d_model"] // cfg["num_heads"] if cfg else None,
    }


@app.get("/api/metrics")
def api_metrics(max_points: int = 400):
    """Training curves from metrics.csv, downsampled for plotting."""
    path = STATE["metrics_path"]
    if not os.path.exists(path):
        return {"available": False, "rows": [], "path": path}

    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows.append({
                    "step": int(row["step"]),
                    "loss": float(row["loss"]),
                    "perplexity": float(row["perplexity"]),
                    "tokens_per_sec": float(row["tokens_per_sec"]),
                    "elapsed_sec": float(row["elapsed_sec"]),
                })
            except (KeyError, ValueError):
                continue  # skip a partially-flushed final row

    total = len(rows)
    if total > max_points:
        stride = total // max_points + 1
        rows = rows[::stride] + [rows[-1]]

    summary = {}
    if rows:
        losses = [r["loss"] for r in rows]
        summary = {
            "steps": total,
            "final_loss": rows[-1]["loss"],
            "best_loss": min(losses),
            "final_perplexity": rows[-1]["perplexity"],
            "elapsed_sec": rows[-1]["elapsed_sec"],
            "mean_tokens_per_sec": sum(r["tokens_per_sec"] for r in rows) / len(rows),
        }

    return {"available": bool(rows), "rows": rows, "summary": summary, "path": path}


@app.post("/api/chat")
def api_chat(req: ChatRequest):
    """Stream a completion as SSE. Each event is {"token": "..."} or {"done": ...}."""

    def events():
        if STATE["model"] is None:
            payload = {"error": STATE["error"] or "no model loaded"}
            yield f"data: {json.dumps(payload)}\n\n"
            return

        model, tokenizer = STATE["model"], STATE["tokenizer"]
        prompt = render_chat_prompt(req.messages, system=req.system)

        n, t0 = 0, time.time()
        try:
            for delta in generate_stream(
                model, tokenizer, prompt,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                max_len=STATE["config"]["max_len"],
                seed=req.seed,
            ):
                n += 1
                yield f"data: {json.dumps({'token': delta})}\n\n"
        except Exception as e:  # surface generation failures in the UI, don't hang the stream
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return

        elapsed = time.time() - t0
        rate = n / elapsed if elapsed > 0 else 0.0
        STATE["tokens_per_sec"] = rate
        yield f"data: {json.dumps({'done': True, 'tokens': n, 'elapsed': elapsed, 'tokens_per_sec': rate})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default=os.environ.get("TRANSFORMER_CKPT", "checkpoints/latest.ckpt"),
                        help="CUDA .ckpt or train.py .npz checkpoint")
    parser.add_argument("--tokenizer", choices=["bbpe", "byte"], default="byte")
    parser.add_argument("--tokenizer-path", default="tokenizer/tok_out/tokenizer.bbpe")
    parser.add_argument("--vocab-size", type=int, default=32000, help="only used with --tokenizer bbpe")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="only used for .npz checkpoints, which don't record it")
    parser.add_argument("--metrics-path", default="metrics.csv")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    STATE["metrics_path"] = args.metrics_path
    load_model(args.ckpt, args.tokenizer, args.tokenizer_path, args.num_heads, args.vocab_size)

    import uvicorn
    print(f"[serve] http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
