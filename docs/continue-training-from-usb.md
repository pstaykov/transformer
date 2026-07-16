# Continuing training on another machine from a USB drive

This covers moving a checkpoint (plus the tokenizer and corpus it depends on) to another
machine via USB, and resuming CUDA training there.

## 1. What to copy onto the USB drive

Three things — none of them are in git:

| File | Size | Why it's not in git |
|---|---|---|
| `cuda/checkpoints/run3/latest.ckpt` (or any `ckpt_stepN.ckpt`) | ~317MB | Checkpoints are gitignored (`checkpoints/`) |
| `tokenizer/tok_out_kevindata/tokenizer.bbpe` | ~254KB | Tokenizer output is gitignored |
| `KEVINDATA/mein_trainingsdaten_10gb.txt` | ~10GB | Raw corpus is gitignored (`/KEVINDATA`) |

```bash
# from the repo root, with the USB drive mounted at /media/<user>/<label>
USB=/media/pstay/<label>
mkdir -p "$USB/kevin_transfer"
cp cuda/checkpoints/run3/latest.ckpt "$USB/kevin_transfer/"
cp tokenizer/tok_out_kevindata/tokenizer.bbpe "$USB/kevin_transfer/"
cp KEVINDATA/mein_trainingsdaten_10gb.txt "$USB/kevin_transfer/"
```

Safely unplug the drive before moving it (`udisksctl unmount -b /dev/sdb1` or use the
desktop's "Eject" action) — pulling it out mid-write on a large file corrupts it.

## 2. On the other machine: get the code

```bash
git clone https://github.com/pstaykov/transformer.git
cd transformer
```

## 3. Copy the three files off the USB drive into place

```bash
USB=/media/<user>/<label>          # wherever it mounts on this machine
mkdir -p cuda/checkpoints/run3 tokenizer/tok_out_kevindata KEVINDATA

cp "$USB/kevin_transfer/latest.ckpt"      cuda/checkpoints/run3/
cp "$USB/kevin_transfer/tokenizer.bbpe"   tokenizer/tok_out_kevindata/
cp "$USB/kevin_transfer/mein_trainingsdaten_10gb.txt" KEVINDATA/
```

## 4. Build the CUDA trainer

Needs: an NVIDIA GPU + driver, the CUDA toolkit (`nvcc`), `cmake`, a C++17 compiler.

```bash
cd cuda
mkdir build && cd build
cmake ..
make -j
```

If `cmake` fails on `find_package(CUDAToolkit REQUIRED)`, the CUDA toolkit isn't
installed/on `PATH` on this machine — install it first (`nvidia-cuda-toolkit` or NVIDIA's
own installer, matching your driver version).

## 5. Resume training

From `cuda/`:

```bash
./build/train_transformer_cuda \
  --resume checkpoints/run3/latest.ckpt \
  --corpus ../KEVINDATA/mein_trainingsdaten_10gb.txt \
  --tokenizer bbpe --tokenizer-path ../tokenizer/tok_out_kevindata/tokenizer.bbpe \
  --batch-size 16 --steps 300000 --lr 1e-4 --min-lr 1e-5 --warmup-steps 500 \
  --grad-clip 1.0 --label-smoothing 0.05 --dropout 0.1 \
  --log-every 50 --checkpoint-every 2000 \
  --checkpoint-dir checkpoints/run3 --metrics-path checkpoints/run3/metrics.csv
```

Notes:

- Architecture (`d_model`, layers, heads, `d_ff`, vocab size) is read straight from the
  checkpoint header — you don't need to (and can't) override it via CLI flags when
  `--resume` is set.
- `--reset-step` is **not** passed, so the step counter continues from wherever the
  checkpoint left off (absolute step numbers keep climbing, not restarting at 0). Only use
  `--reset-step` for a genuinely new training phase (e.g. SFT on different data).
- `--lr`/`--min-lr`/`--warmup-steps` here are a deliberately gentle "warm restart" (low
  peak, short warmup) rather than repeating the original run's full schedule
  (`--lr 3e-4 --warmup-steps 2000`) — jumping straight back to a high peak LR right after
  a model has converged tends to spike the loss before it re-settles. Adjust `--steps` to
  however far you want this phase to run.
- `--batch-size 16` assumes similar GPU memory to the original training machine. Lower it
  if you hit an out-of-memory error on the new machine's GPU.

## 6. If you're running this at the same time as another machine

Training two independent continuations from the same checkpoint (e.g. one machine still
running locally, another resuming from the USB copy) creates two diverging branches, not
a synced/distributed job — they don't merge back together automatically. Use different
`--checkpoint-dir` values if you want to keep both histories, and decide later which
checkpoint lineage to keep going with.
