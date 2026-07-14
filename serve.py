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
import threading
import time

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

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

# Transformer.forward() (model.py) stashes intermediate activations on `self`
# (e.g. self._ln_f_out, self._W_out_q), and the per-layer blocks do the same
# internally. There is no KV cache, so a full generation is many forward()
# calls in a row. FastAPI runs sync endpoints (like api_chat's `events()`
# generator) in a threadpool, so two simultaneous POST /api/chat requests
# would otherwise call forward() on the *same* Transformer instance from two
# threads at once and clobber each other's activations mid-generation,
# producing garbled logits/output (not just a race on paper - it's an actual
# shared mutable attribute). GENERATION_LOCK serializes generation so
# concurrent requests queue instead of interleaving.
GENERATION_LOCK = threading.Lock()

# Bound how many requests may be queued (waiting on GENERATION_LOCK or already
# generating) at once. Without this, N idle browser tabs (or one script firing
# many requests) can each spawn a threadpool worker that blocks forever on the
# lock, exhausting the pool and making the server unresponsive to everyone.
# Generation is fully serialized above, so "concurrent" generation is really
# just queue depth; 4 is enough slack for a couple of LAN users without
# letting the queue grow unbounded.
MAX_QUEUED_REQUESTS = 4
_queue_count = 0
_queue_count_lock = threading.Lock()

# Input clamps: a client (or a bug in the UI) can send arbitrary values. These
# are sized off the model's actual context window (max_len, 256 tokens on
# every checkpoint this project has trained) rather than picked arbitrarily -
# KEVIN re-reads its *entire* context on every token (no KV cache), so
# anything beyond max_len is silently truncated by generate_stream anyway.
# Accepting more than that just wastes an LAN client's/attacker's request for
# nothing, so the caps track the real budget instead of an arbitrary "big
# enough" number.
def _context_len():
    cfg = STATE.get("config")
    return cfg["max_len"] if cfg else 256


def _max_new_tokens_cap():
    return max(16, _context_len() // 2)  # leave half the window for the prompt


MAX_MESSAGES = 8           # a long back-and-forth won't fit in 256 tokens anyway
MAX_MESSAGE_CHARS = 240    # ~roughly 256 tokens' worth of BPE text, generously

# ---------------------------------------------------------------------------
# Zero-trust network layer: every request is treated as untrusted, including
# ones from inside the LAN - a rate-limited or banned IP is rejected the same
# way whether it's a stranger who found the port or a misbehaving device on
# the same network. This is on top of, not instead of, the input clamps above
# and the generation lock/queue cap - defense in depth.
# ---------------------------------------------------------------------------

_SECURITY_LOCK = threading.Lock()
_request_log = {}   # ip -> sorted list of recent request timestamps (all routes)
_chat_log = {}       # ip -> same, but only for POST /api/chat (the expensive route)
_violations = {}     # ip -> count of rate-limit/oversize-body violations
_banned_until = {}   # ip -> unix timestamp its ban expires

RATE_LIMIT_WINDOW_SEC = 60
RATE_LIMIT_MAX_GENERAL = 60   # page loads + polling /api/model, /api/metrics
RATE_LIMIT_MAX_CHAT = 8       # generation is expensive; keep this tight
MAX_BODY_BYTES = 16 * 1024    # a legitimate chat request is a few KB at most
VIOLATIONS_BEFORE_BAN = 5
BAN_DURATION_SEC = 15 * 60
_IP_TRACKING_CAP = 2000       # prune stale IPs past this so uptime can't leak memory


def _prune_window(log, now):
    cutoff = now - RATE_LIMIT_WINDOW_SEC
    while log and log[0] < cutoff:
        log.pop(0)


def _record_violation(ip, now):
    count = _violations.get(ip, 0) + 1
    _violations[ip] = count
    if count >= VIOLATIONS_BEFORE_BAN:
        _banned_until[ip] = now + BAN_DURATION_SEC
        print(f"[serve] banning {ip} for {BAN_DURATION_SEC}s after {count} violations")


def _prune_stale_ips(now):
    if len(_request_log) + len(_chat_log) <= _IP_TRACKING_CAP:
        return
    stale_cutoff = now - RATE_LIMIT_WINDOW_SEC * 10
    for log in (_request_log, _chat_log):
        for ip in [ip for ip, ts in log.items() if not ts or ts[-1] < stale_cutoff]:
            del log[ip]


@app.middleware("http")
async def zero_trust_gate(request: Request, call_next):
    ip = request.client.host if request.client else "unknown"
    now = time.time()

    with _SECURITY_LOCK:
        ban_until = _banned_until.get(ip)
        if ban_until is not None:
            if now < ban_until:
                return JSONResponse(
                    {"error": f"banned for {int(ban_until - now)}s after repeated abuse"},
                    status_code=403,
                )
            del _banned_until[ip]  # ban expired: clean slate
            _violations.pop(ip, None)

        content_length = request.headers.get("content-length")
        if content_length is not None and int(content_length) > MAX_BODY_BYTES:
            _record_violation(ip, now)
            return JSONResponse({"error": "request body too large"}, status_code=413)

        is_chat = request.url.path == "/api/chat"
        log = _chat_log.setdefault(ip, []) if is_chat else _request_log.setdefault(ip, [])
        limit = RATE_LIMIT_MAX_CHAT if is_chat else RATE_LIMIT_MAX_GENERAL
        _prune_window(log, now)

        if len(log) >= limit:
            _record_violation(ip, now)
            return JSONResponse(
                {"error": "rate limit exceeded, slow down"},
                status_code=429,
                headers={"Retry-After": str(RATE_LIMIT_WINDOW_SEC)},
            )

        log.append(now)
        _prune_stale_ips(now)

    return await call_next(request)


class ChatRequest(BaseModel):
    messages: list[dict]
    max_new_tokens: int = 96
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    system: str | None = None
    seed: int | None = None

    @field_validator("messages")
    @classmethod
    def _clamp_messages(cls, v):
        if len(v) > MAX_MESSAGES:
            v = v[-MAX_MESSAGES:]
        for m in v:
            if not isinstance(m, dict) or "role" not in m or "content" not in m:
                raise ValueError("each message needs a 'role' and 'content'")
            m["content"] = str(m["content"])[:MAX_MESSAGE_CHARS]
        return v

    @field_validator("max_new_tokens")
    @classmethod
    def _clamp_max_new_tokens(cls, v):
        return max(1, min(int(v), _max_new_tokens_cap()))

    @field_validator("temperature")
    @classmethod
    def _clamp_temperature(cls, v):
        return max(0.0, min(float(v), 5.0))

    @field_validator("top_k")
    @classmethod
    def _clamp_top_k(cls, v):
        return max(0, min(int(v), 1000))

    @field_validator("top_p")
    @classmethod
    def _clamp_top_p(cls, v):
        return max(0.0, min(float(v), 1.0))

    @field_validator("system")
    @classmethod
    def _clamp_system(cls, v):
        return v[:MAX_MESSAGE_CHARS] if v else v


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
        global _queue_count

        if STATE["model"] is None:
            payload = {"error": STATE["error"] or "no model loaded"}
            yield f"data: {json.dumps(payload)}\n\n"
            return

        # Reserve a queue slot up front so a burst of requests can't all pile
        # up waiting on GENERATION_LOCK; reject past MAX_QUEUED_REQUESTS with
        # a clear error instead of leaving the client's connection hanging.
        with _queue_count_lock:
            if _queue_count >= MAX_QUEUED_REQUESTS:
                yield f"data: {json.dumps({'error': 'server busy, try again'})}\n\n"
                return
            _queue_count += 1

        try:
            model, tokenizer = STATE["model"], STATE["tokenizer"]
            n, t0 = 0, time.time()
            try:
                with GENERATION_LOCK:
                    prompt = render_chat_prompt(req.messages, system=req.system)
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
            except Exception as e:
                # Catches failures anywhere in this block - bad prompt data,
                # tokenizer errors, OOM, etc. - not just ones raised inside
                # generate_stream, so a single bad request can't take the
                # server process down.
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                return

            elapsed = time.time() - t0
            rate = n / elapsed if elapsed > 0 else 0.0
            STATE["tokens_per_sec"] = rate
            yield f"data: {json.dumps({'done': True, 'tokens': n, 'elapsed': elapsed, 'tokens_per_sec': rate})}\n\n"
        finally:
            with _queue_count_lock:
                _queue_count -= 1

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
