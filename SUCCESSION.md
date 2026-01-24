# Succession Notes for Next Agent

*Last updated: 2026-01-23 (Session 9)*

## Current Status

**PHASE**: Training run in progress - monitoring needed

**WHAT'S RUNNING NOW**:
- Training: `src/training/pipelined_trainer_v2.py`
- Data: `data/oracle_generated/judgment_v2_train.jsonl` (896 examples)
- Output: `models/judgment_v2_full/`
- Config: batch=3, gens=5, lr=2e-6, steps=1000, save_steps=25
- Resuming from: checkpoint-50
- Gemini: max_concurrent=8 (dynamic backoff on 429)
- Max completion length: 1536 tokens (fixed from 768)

**SESSION 9 COMPLETED**:
1. Fixed JSON parse errors: `max_output_tokens` 8192 → 16384
2. Increased Gemini concurrency: 5 → 8 (with dynamic 429 backoff)
3. Fixed max_completion_length: 768 → 1536
4. Unified structured logging - all logs now flow through proper channels
5. Added full content logging: `logs/runs/*/content/{inference_id}_full.txt`
6. Cleaned up orphaned training logs and redundant run-level log files
7. Archived retired trainers to `src/training/_archived/`
8. Updated heartbeat with training run prohibition

## Logging Architecture (Clean)

```
logs/
├── gemini_*.log           # Session-level Gemini API calls (active)
├── evaluation_*.log       # Session-level evaluation parsing (active)
├── current -> runs/{id}   # Symlink to active run
└── runs/{run_id}/
    ├── config.json        # Run configuration snapshot
    ├── training.log       # Structured training events with [run=X step=Y]
    ├── metrics.jsonl      # Per-step structured metrics
    ├── inferences.jsonl   # Individual inference records with content_path
    ├── summary.json       # Written on run completion/interrupt
    ├── checkpoints.json   # Checkpoint manifest
    ├── parent_run.txt     # Link to parent run if resumed
    └── content/           # Full prompt/completion/response text
        ├── {inf_id}_full.txt  # Local: prompt + completion
        └── {inf_id}_full.txt  # Remote: gemini prompt + response
```

**Inference ID format**: `{run_suffix}.{step}.{L|R}.{idx}`
- L = Local (model generation)
- R = Remote (Gemini evaluation)

## Quick Commands

```bash
# Activate environment
cd /home/dgk/projects/cognitiveop_attunement
source .venv/bin/activate

# Watch live training
tail -f logs/current/training.log

# Check structured metrics
cat logs/current/metrics.jsonl | tail -5

# Check inferences (with content paths)
cat logs/current/inferences.jsonl | tail -10

# View a specific completion/response
cat logs/current/content/aining.51.L.0_full.txt

# Check Gemini concurrency and rate limits
grep -i "max_concurrent\|RATE_LIMITED" logs/gemini_*.log | tail -20

# View run summary (after completion/interrupt)
cat logs/current/summary.json

# Check GPU
nvidia-smi --query-gpu=memory.used --format=csv

# Run tests
pytest tests/ -v --tb=short
```

## Rate Limit Handling

Gemini API calls now have dynamic backoff:
- Default: 8 concurrent calls
- On 429 error: reduces concurrency by 1 (min 1), waits 30 seconds
- Logged with `[RATE_LIMITED]` tag in gemini log

## 🎯 NEXT PRIORITY: Dashboard

**Task**: Update `scripts/training_dashboard.py` for new logging paradigm

**Required features**:
1. **Run selector**: List all runs from `logs/runs/`, show status
2. **Metrics visualization**: 
   - Loss, reward_mean/std, grad_norm over time
   - Correct count trend
3. **Inference drill-down**:
   - View individual completions for a step
   - See 8-dimension Gemini scores
   - Link to content files
4. **Run chain view**: Show continuation chain for resumed runs

**Data sources**:
- `logs/runs/*/metrics.jsonl` → structured step metrics
- `logs/runs/*/inferences.jsonl` → inference records with content_path
- `logs/runs/*/content/*.txt` → full prompt/completion/response text
- `logs/runs/*/summary.json` → run metadata

## Files Modified This Session

| File | Changes |
|------|---------|
| `src/evaluation/async_reward.py` | max_output_tokens 16384, max_concurrent 8, 429 backoff |
| `src/evaluation/logging_config.py` | Removed get_training_logger() |
| `src/training/pipelined_trainer_v2.py` | Unified log(), content saving, max_completion_length 1536 |
| `src/logging/run_logger.py` | save_content(), removed gemini/evaluation logs, content_path field |
| `.gitignore` | Added logs/ |
| `CLAUDE.md` | Added training run prohibition to heartbeat |

**Archived**: `src/training/_archived/` contains retired pipelined_trainer.py and pipelined_grpo_trainer.py

## What NOT to Do

1. **Don't launch training runs** - provide commands for user to run in separate terminal
2. **Don't modify llm_evaluator.py system instruction** - would invalidate Gemini cache
3. **Don't start another GPU process** - training is using it

## If Training Needs Restart

```bash
source .venv/bin/activate && CUDA_VISIBLE_DEVICES=0 python -c "
from src.training.pipelined_trainer_v2 import train_pipelined_v2
train_pipelined_v2(
    data_path='data/oracle_generated/judgment_v2_train.jsonl',
    output_dir='models/judgment_v2_full',
    num_steps=1000,
    batch_size=3,
    num_generations=5,
    learning_rate=2e-6,
    warmup_ratio=0.1,
    save_steps=25,
    log_level='debug',
    gemini_max_concurrent=8,
    resume_from_checkpoint='models/judgment_v2_full/checkpoint-75',  # Update to latest
)
"
```
