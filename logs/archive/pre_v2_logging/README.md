# Pre-V2 Logging Archive

Archived: 2026-01-23

## Contents

### 20260123_152241_judgment_v2_legacy/
First 50 steps of judgment training (the only substantive data):
- `training.log` - 1138 lines of training output  
- `summary.json` - Run metadata (status: interrupted, 50 steps, ~2.5 hours)
- `checkpoints.json` - checkpoint-25, checkpoint-50
- `config.json` - Training configuration

Key metrics:
- Steps trained: 50
- Duration: ~2.5 hours (8839 seconds)
- Final reward mean: 0.71
- Final loss: -0.04

### 20260123_221532_training/
Aborted continuation attempt (steps 51-52), superseded by new logging paradigm.
- `metrics.jsonl` - 1 step of metrics

## What Was Removed
Trivial debugging session logs (1-5 lines each, just initialization messages):
- `evaluation_*.log`, `gemini_*.log`, `training_*.log`
- `index.json` (run index, now stale)
