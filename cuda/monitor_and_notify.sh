#!/usr/bin/env bash
# Periodically emails a KEVIN training status update (samples + loss/ppl chart)
# while cuda/checkpoints/run3's training process is running, and restarts
# training if it dies unexpectedly. Only intended to run from repo root's
# cuda/ directory. Not meant to be edited/extended - a one-off babysitter for
# the current continued-training run.
set -u
cd "$(dirname "$0")"
source ../.venv/bin/activate

RUN_DIR=checkpoints/run3
PIDFILE=$RUN_DIR/train.pid
LOGFILE=$RUN_DIR/train.log
INTERVAL=${1:-1800}

train_alive() {
  [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null
}

restart_training() {
  echo "$(date -Is) monitor: training not running, resuming from $RUN_DIR/latest.ckpt" >> "$LOGFILE"
  nohup ./build/train_transformer_cuda \
    --resume $RUN_DIR/latest.ckpt \
    --corpus ../KEVINDATA/mein_trainingsdaten_10gb.txt \
    --tokenizer bbpe --tokenizer-path ../tokenizer/tok_out_kevindata/tokenizer.bbpe \
    --batch-size 16 --steps 300000 --lr 1e-4 --min-lr 1e-5 --warmup-steps 500 \
    --grad-clip 1.0 --label-smoothing 0.05 --dropout 0.1 \
    --log-every 50 --checkpoint-every 2000 \
    --checkpoint-dir $RUN_DIR --metrics-path $RUN_DIR/metrics.csv \
    >> "$LOGFILE" 2>&1 < /dev/null &
  disown
  echo $! > "$PIDFILE"
}

while true; do
  sleep "$INTERVAL"

  if ! train_alive; then
    restart_training
    sleep 120
  fi

  step=$(tail -n 200 "$LOGFILE" | grep -oE 'step +[0-9]+' | tail -1 | grep -oE '[0-9]+')
  status=$(train_alive && echo "running" || echo "DOWN")

  samples_json=$(cd .. && python3 - "$RUN_DIR/latest.ckpt" <<'PYEOF'
import sys, subprocess, json, ast
ckpt = sys.argv[1]
prompts = ["Hello, I'm Kevin and", "The"]
out = []
for p in prompts:
    r = subprocess.run(
        ["python3", "generate.py", "--resume", ckpt, "--prompt", p,
         "--tokenizer", "bbpe", "--tokenizer-path", "tokenizer/tok_out_kevindata/tokenizer.bbpe",
         "--num-tokens", "30"],
        capture_output=True, text=True, timeout=180
    )
    completion = ""
    for line in r.stdout.splitlines():
        if line.startswith("decoded"):
            raw = line.split(":", 1)[-1].strip()
            try:
                completion = ast.literal_eval(raw)
            except Exception:
                completion = raw
    out.append({"prompt": p, "completion": completion or ("(generation failed) " + r.stderr[-300:])})
print(json.dumps(out))
PYEOF
)

  cd ..
  python3 - "$step" "$status" "$samples_json" <<'PYEOF'
import sys, json
from utils import notify

step_s, status, samples_json = sys.argv[1], sys.argv[2], sys.argv[3]
step = int(step_s) if step_s else None
samples = [(s["prompt"], s["completion"]) for s in json.loads(samples_json)]

config = notify.load_env_file("email.env")
note = f"cuda trainer status: {status}, continuing on the big corpus (mein_trainingsdaten_10gb.txt)."
notify.send_note(note, config, step=step, metrics_path="cuda/checkpoints/run3/metrics.csv", samples=samples)
print("status email sent.")
PYEOF
  cd cuda
done
