# Succession Notes for Next Agent

*Last updated: 2026-01-23 (Session 8)*

## Immediate Context

**PHASE**: Training run in progress with PipelinedTrainerV2

**WHAT'S RUNNING NOW**:
- Training: `src/training/pipelined_trainer_v2.py`
- Data: `data/oracle_generated/judgment_v2_train.jsonl` (896 examples)
- Output: `models/judgment_v2_full/`
- Config: batch=3, gens=5, lr=2e-6, steps=1000, save_steps=25
- Resuming from: checkpoint-50

**WHAT JUST HAPPENED (Session 8)**:
1. Implemented structured run logging (`src/logging/run_logger.py`)
2. Added parallel Gemini evaluation (independent calls, not batch)
3. Comprehensive ID hierarchy: run_id → step_id → inference_id
4. JSONL structured logs: `metrics.jsonl`, `inferences.jsonl`
5. Resilience features: SIGTERM/SIGINT handlers, atomic checkpoints, atexit cleanup
6. Checkpoint inheritance across run continuation chain
7. Archived pre-v2 logs to `logs/archive/pre_v2_logging/`

## New Logging Paradigm

**Directory structure**:
```
logs/
├── current -> runs/{active_run_id}  (symlink)
├── runs/
│   └── {YYYYMMDD_HHMMSS}_{experiment}/
│       ├── config.json          # Run configuration snapshot
│       ├── training.log         # Human-readable log
│       ├── gemini.log          # Gemini API calls
│       ├── evaluation.log      # Evaluation details
│       ├── metrics.jsonl       # Per-step structured metrics
│       ├── inferences.jsonl    # Individual inference records
│       ├── summary.json        # Written on run completion
│       ├── checkpoints.json    # Checkpoint manifest
│       └── parent_run.txt      # Link to parent run if resumed
└── archive/                     # Historical runs
```

**Inference ID format**: `{run_suffix}.{step}.{L|R}.{idx}`
- L = Local (model generation)
- R = Remote (Gemini evaluation)
- Example: `aining.51.L.3` = run ending "aining", step 51, local, index 3

**Run chain**: When resuming, inherits checkpoints from parent runs. `checkpoints_saved` in summary.json includes both inherited and new checkpoints.

## Quick Commands

```bash
# Activate environment
cd /home/dgk/projects/cognitiveop_attunement
source .venv/bin/activate

# Watch live training
tail -f logs/current/training.log

# Check structured metrics
cat logs/current/metrics.jsonl | tail -5

# Check inferences
cat logs/current/inferences.jsonl | tail -10

# View run summary (after completion/interrupt)
cat logs/current/summary.json

# Check GPU
nvidia-smi --query-gpu=memory.used --format=csv

# Run tests
pytest tests/ -v --tb=short
```

## Monitoring the Training Run

**Key metrics to watch** (from `metrics.jsonl`):
- `grad_norm`: Should be 1-10 range (was ~250 before normalization fix)
- `loss`: Trending toward 0 (negative is normal for GRPO)
- `reward_mean`: Should gradually increase
- `correct_count/total_count`: Accuracy per batch

**Inference tracking** (from `inferences.jsonl`):
- `inference_type: "local"` - Model generations (token_count, char_count)
- `inference_type: "remote"` - Gemini evals (scores on 8 dimensions)

## Resilience Features

1. **Graceful shutdown**: Ctrl+C or SIGTERM triggers emergency checkpoint + clean finalization
2. **Atomic checkpoints**: Saves to `.tmp_checkpoint-N` then renames (prevents corruption)
3. **Atexit cleanup**: Backup cleanup for unexpected exits
4. **Double-finalization guard**: Safe to call finalize() multiple times

## Dashboard Requirements

**To build**: `scripts/training_dashboard.py` needs updating for new logging paradigm

**Required features**:
1. **Run selector**: List all runs from `logs/runs/`, show status (completed/interrupted/in_progress)
2. **Metrics visualization**: 
   - Loss over time
   - Reward mean/std over time
   - grad_norm over time
   - Correct count trend
3. **Inference drill-down**:
   - View individual completions for a step
   - See 8-dimension scores from Gemini
   - Compare across completions
4. **Run chain view**: Show continuation chain for resumed runs

**Data sources**:
- `logs/current/` symlink → active run
- `logs/runs/*/metrics.jsonl` → structured step metrics
- `logs/runs/*/inferences.jsonl` → individual inference records
- `logs/runs/*/summary.json` → run metadata and final state

**Note on "run" vs checkpoints**: A "run" is a single execution session (from start to interrupt/completion). Checkpoints are snapshots of model weights within or across runs. A resumed run inherits checkpoints from its parent run chain but is itself a distinct run.

## Files Modified This Session

| File | Changes |
|------|---------|
| `src/logging/__init__.py` | New module exports |
| `src/logging/run_logger.py` | RunLogger, StepMetrics, InferenceRecord, ThreadSafeJSONLWriter |
| `src/evaluation/async_reward.py` | Parallel Gemini calls, run_id/run_logger integration |
| `src/evaluation/async_evaluator.py` | Individual prompts for parallel eval |
| `src/evaluation/llm_evaluator.py` | Single completion evaluation support |
| `src/evaluation/logging_config.py` | QueueHandler setup for async-safe logging |
| `src/training/pipelined_trainer_v2.py` | Signal handlers, atexit, atomic checkpoints, RunLogger integration |

## What NOT to Do

1. **Don't kill the training run** unless clearly broken
2. **Don't modify llm_evaluator.py** while training - would invalidate the Gemini cache
3. **Don't start another training run** - GPU is occupied

## If Training Crashes

1. Check `logs/current/summary.json` - should have `status: "failed"` or `"interrupted"`
2. Check `logs/current/training.log` for error details
3. Find last checkpoint in `models/judgment_v2_full/checkpoint-*`
4. Can resume with `resume_from_checkpoint='models/judgment_v2_full/checkpoint-N'`

## Next Steps

1. **Monitor current run** - watch metrics, ensure stability
2. **Build dashboard** - visualize runs with new logging format
3. **After run completes**: Evaluate trained adapter on validation set
4. **Compare to baseline**: Does trained model show improvement on REVERSION?
