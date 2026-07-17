# tools/

Standalone maintenance scripts. None of these are needed to train or run
inference day-to-day (that's `./setup.sh`, `./train.sh`, `./infer.sh` at the
repo root) — they're for setup, dataset prep, and monitoring. Run all of them
from the repo root, e.g. `python tools/download_models.py`.

| Script | What it does |
| --- | --- |
| `download_models.py` | Pulls the pretrained base + chat checkpoints and tokenizer from Hugging Face into `cuda/checkpoints/` and `tokenizer/tok_out_kevindata/`. Called by `./setup.sh` and `./setup-docker.sh`; run it directly to re-download or refresh (`--force`). |
| `convert_everyday_conversations.py` | Downloads `HuggingFaceTB/everyday-conversations-llama3.1-2k` and writes it out as the JSONL chat format the trainers expect. This is how `data/everyday_conversations.jsonl` was generated — re-run it only if you want to refresh that file. |
| `gen_evolution.py` | Loads a handful of checkpoints spanning a training run and generates sample completions at each, writing `web/data/evolution.json` for the showcase site's "how the model talked over time" section. Offline/one-off — too slow to do live in `serve.py`. |
| `send_note.py` | Sends a one-off "training status" email (loss chart + sample completions) using `email.env` credentials. Handy for a manual check-in on a long run; see `docs/monitoring.md` for the automated version. |
