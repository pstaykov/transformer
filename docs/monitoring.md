# Babysitting a long training run

Both scripts below live in `cuda/` and are meant to be left running (e.g. in
`tmux`/`nohup`) alongside a long CUDA training run, restarting the trainer if
it dies. Neither is meant to be edited/extended — they're one-off babysitters
tied to a specific run's paths and hyperparameters, not general tooling. Copy
and adapt one if you need to babysit a different run.

## `monitor_train.sh` — restart-on-death, local logging only

```bash
cd cuda
./monitor_train.sh [interval_seconds]   # default 300s
```

Polls every `interval_seconds` whether the trainer process (tracked via a
`.pid` file under its checkpoint dir) is still alive. If it died:

- Checks the tail of the training log for an out-of-memory error. If found,
  halves `--batch-size` (down to a floor of 2) before restarting — otherwise
  restarts as-is.
- Resumes from `checkpoint-dir/latest.ckpt` with `nohup`, logs everything to
  `monitor.log` under the checkpoint dir.

No email or other side effects outside that directory — check `monitor.log`
for history.

## `monitor_and_notify.sh` — restart-on-death + periodic status emails

```bash
cd cuda
./monitor_and_notify.sh [interval_seconds]   # default 1800s
```

Same restart-on-death idea, but additionally emails a status update (via
`utils/notify.py` and `email.env` credentials) every interval: current step,
loss/perplexity chart, and a couple of live sample completions generated with
`generate.py`. Useful when you're not at a terminal to check `monitor_train.sh`'s
log yourself.

## Checking on either one

```bash
ps aux | grep monitor_train        # or monitor_and_notify
tail -f cuda/checkpoints/<run>/monitor.log   # monitor_train.sh only
```

To stop a monitor loop, kill its shell process — this does **not** stop the
training process it's babysitting (that's tracked separately via the `.pid`
file), so kill that too if you want training to actually stop.
