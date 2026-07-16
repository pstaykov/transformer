#!/usr/bin/env bash
# Babysits the sft_wildchat training run: checks periodically that the
# trainer process is still alive, and if it has died, inspects the tail of
# its log to figure out why and restarts it - halving --batch-size on an
# out-of-memory death (down to a floor), otherwise just resuming as-is.
# No email/notification side effects - local logging only.
set -u
cd "$(dirname "$0")"
source ../.venv/bin/activate

RUN_DIR=checkpoints/sft_wildchat
PIDFILE=$RUN_DIR/train.pid
LOGFILE=$RUN_DIR/train.log
MONITOR_LOG=$RUN_DIR/monitor.log
BATCH_FILE=$RUN_DIR/monitor_batch_size
INTERVAL=${1:-300}
MIN_BATCH=2

[ -f "$BATCH_FILE" ] || echo 12 > "$BATCH_FILE"

log() { echo "$(date -Is) monitor: $*" >> "$MONITOR_LOG"; }

train_alive() {
  [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null
}

last_step() {
  tail -n 500 "$LOGFILE" 2>/dev/null | grep -oE 'step +[0-9]+' | tail -1 | grep -oE '[0-9]+'
}

died_of_oom() {
  tail -n 50 "$LOGFILE" 2>/dev/null | grep -qi "out of memory"
}

restart_training() {
  local batch
  batch=$(cat "$BATCH_FILE")

  if died_of_oom; then
    local new_batch=$(( batch / 2 ))
    if [ "$new_batch" -lt "$MIN_BATCH" ]; then new_batch=$MIN_BATCH; fi
    if [ "$new_batch" -ne "$batch" ]; then
      log "last death was OOM at batch-size $batch; lowering to $new_batch and retrying"
      batch=$new_batch
      echo "$batch" > "$BATCH_FILE"
    else
      log "last death was OOM but batch-size already at floor ($batch); retrying as-is"
    fi
  else
    log "training not running (no OOM detected in recent log); resuming at batch-size $batch from $RUN_DIR/latest.ckpt"
  fi

  nohup ./build/train_transformer_cuda \
    --resume $RUN_DIR/latest.ckpt \
    --corpus ../data/wildchat_en.jsonl --data-format chat \
    --tokenizer bbpe --tokenizer-path ../tokenizer/tok_out_kevindata/tokenizer.bbpe \
    --batch-size "$batch" --lr 5e-5 --min-lr 5e-6 --steps 196000 --warmup-steps 500 \
    --grad-clip 1.0 --label-smoothing 0.05 --dropout 0.1 \
    --log-every 50 --checkpoint-every 10000 \
    --checkpoint-dir $RUN_DIR --metrics-path $RUN_DIR/metrics.csv \
    >> "$LOGFILE" 2>&1 < /dev/null &
  disown
  echo $! > "$PIDFILE"
  log "restarted, new pid $!"
}

log "monitor started (interval=${INTERVAL}s, pid $$)"

while true; do
  sleep "$INTERVAL"

  if ! train_alive; then
    log "detected training process down (last step: $(last_step))"
    restart_training
    sleep 120
    if train_alive; then
      log "restart looks healthy (pid $(cat "$PIDFILE"))"
    else
      log "restart failed to stay up - will retry again next interval"
    fi
    continue
  fi

  log "alive, pid $(cat "$PIDFILE"), last step $(last_step)"
done
