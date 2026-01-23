# Succession Notes for Next Agent

*Last updated: 2026-01-23 (Session 7)*

## Immediate Context

**PHASE**: 1000-step training run in progress with PipelinedTrainerV2

**WHAT'S RUNNING NOW**:
- Training: `src/training/pipelined_trainer_v2.py`
- Data: `data/oracle_generated/judgment_v2_train.jsonl` (896 examples)
- Output: `models/judgment_v2_full/`
- Config: batch=4, gens=4, lr=2e-6, steps=1000
- Estimated time: ~25 hours total (~90s/step)

**WHAT JUST HAPPENED (Session 7)**:
1. Built `pipelined_trainer_v2.py` - Native model loading (no TRL dependency), cross-step async pipelining
2. Solved training instability: token normalization (0.8 exponent), reduced LR to 2e-6
3. Added CONCISENESS as 7th evaluation dimension
4. **Added Lonergan synopsis to Gemini evaluator** (~2645 tokens in cached system instruction)
5. **Added REVERSION as 8th evaluation dimension** (14% weight) - critical for distinguishing Level 3 from Level 2
6. Created compressed docs (`docs/_compressed/`) for selective system instruction augmentation
7. Conducted signal loss audit comparing early ideal to current implementation

## Training Run Status

**Current metrics (as of step ~14)**:
- grad_norm: ~2.0 (was ~250 before token normalization fix)
- loss: ~-0.17 (negative = favoring high-advantage completions, correct)
- rewards: ~0.26-0.57 range (variance exists, learning possible)
- speed: ~90s/step

**Checkpoints**: Save every 100 steps to `models/judgment_v2_full/checkpoint-{step}/`

## Key Changes This Session

### 1. REVERSION Dimension Added

**Why**: Reversion is THE critical operation distinguishing judgment (Level 3) from understanding (Level 2). Fulfillment is found by reverting from formulation to "the more rudimentary state—to what is merely sensed or merely conscious, not as formulated but as given."

**What it evaluates**:
- High: Student cites SPECIFIC evidence, quotes data, points to particular observations
- Low: Student asserts fulfillment abstractly, manipulates concepts without checking data
- Zero: Pure inference without reversion to data

**New weight distribution** (8 dimensions):
| Dimension | Weight |
|-----------|--------|
| CONDITION_IDENTIFICATION | 14% |
| EVIDENCE_MAPPING | 16% |
| REASONING_VALIDITY | 20% |
| JUDGMENT_COHERENCE | 16% |
| OPERATIONAL_FIDELITY | 12% |
| REVERSION | 14% |
| AUTHENTIC_INTENT | 3% |
| CONCISENESS | 5% |

### 2. Compressed Documentation

Created `docs/_compressed/` with 5 documents (~4400 tokens total):
- `temporal_structure.md` - Temporal reasoning in judgment
- `counterpositions.md` - Detecting self-defeating claims
- `self_appropriation.md` - Philosophic domain, performative consistency
- `interpretation.md` - Expression structure, hermeneutics
- `domain_schema.md` - Mode-of-fulfillment differentiation

See `docs/_compressed/INDEX.md` for mode-specific inclusion recommendations.

### 3. Signal Loss Audit Results

**Restored**: REVERSION (for judgment)

**Deferred to appropriate operations**:
- COMMITMENT → Decision/Responsibility model
- DEVELOPMENTAL_CHARACTER → Creative/Ideational model

## Quick Commands

```bash
# Activate environment
cd /home/dgk/projects/cognitiveop_attunement
source .venv/bin/activate

# Check if training is running
ps aux | grep python | grep train

# Check training progress
tail -50 logs/training_full.log

# Check GPU memory
nvidia-smi --query-gpu=memory.used --format=csv

# Check current step (parse from log)
grep "^Step" logs/training_full.log | tail -5

# Run tests (all 34 evaluation tests should pass)
pytest tests/evaluation/ -v --tb=short
```

## Monitoring the Training Run

**Key things to watch**:

1. **grad_norm**: Should stay in 1-10 range. If it spikes to 100+, training is unstable.

2. **loss**: Should trend toward 0 over time. Negative loss is correct (GRPO favors high-advantage).

3. **reward mean**: Should gradually increase as model improves.

4. **Correct judgments**: Log shows "X/4 correct" per batch. Should trend upward.

5. **VRAM**: Should stay under 20GB. If hitting 24GB, reduce batch size.

**Log locations**:
- `logs/training_full.log` - Main training progress
- `logs/cognitive_eval.log` - Gemini evaluation details

## Files Modified This Session

| File | Changes |
|------|---------|
| `src/evaluation/llm_evaluator.py` | Added REVERSION dimension, Lonergan synopsis, rebalanced weights |
| `src/training/pipelined_trainer_v2.py` | Token normalization (0.8 exp), diagnostic logging |
| `docs/_compressed/*.md` | New compressed documentation |
| `CLAUDE.md` | Updated project state, decisions log |

## What NOT to Do

1. **Don't kill the training run** unless there's a clear problem
2. **Don't modify llm_evaluator.py** while training - would invalidate the cache
3. **Don't start another training run** - GPU is occupied

## If Training Crashes

1. Check the log for error message
2. Note the last checkpoint saved
3. Can resume from checkpoint (not implemented yet, would need to add)
4. Common issues:
   - CUDA OOM: Reduce batch_size or num_generations
   - Gemini rate limit: Add delays or reduce batch size
   - NaN loss: Learning rate too high or gradient explosion

## Next Steps After This Run

1. **Evaluate trained adapter** on held-out validation set
2. **Compare to base model** - does trained model show improvement?
3. **Analyze REVERSION scores** - is model learning to revert to data?
4. **Generate more diverse data** if coverage gaps identified
5. **Train other operations** (Attention, Understanding, Decision)

---

## Continuation Prompt for Next Agent

```
# Task: Analyze Training Run Progress

## Context
A 1000-step training run is in progress using PipelinedTrainerV2. The run trains
Qwen2.5-7B-Instruct to perform Lonergan's cognitive operation of JUDGMENT.

Key recent changes:
- Added REVERSION dimension (14% weight) to evaluator
- Added Lonergan synopsis to Gemini system instruction
- Solved training instability with token normalization

## Your Task

1. **Check training status**:
   ```bash
   ps aux | grep python | grep train
   tail -100 logs/training_full.log
   ```

2. **Analyze metrics**:
   - What step is it on?
   - What's the current grad_norm? (should be 1-10)
   - What's the loss trend?
   - What's the mean reward?
   - How many correct judgments per batch?

3. **Check for problems**:
   - Any error messages?
   - Is grad_norm spiking?
   - Is loss NaN or diverging?
   - Is reward collapsing to 0?

4. **Document findings**:
   - Current step and ETA
   - Key metrics summary
   - Any concerns
   - Recommendations

5. **If training completed**:
   - Check final checkpoint exists
   - Run validation evaluation
   - Compare to baseline

## Key Files

- `logs/training_full.log` - Main training log
- `logs/cognitive_eval.log` - Evaluation details
- `models/judgment_v2_full/` - Output directory
- `src/training/pipelined_trainer_v2.py` - Trainer implementation

## Expected Metrics

- grad_norm: 1-10 (was ~250 before fix)
- loss: trending toward 0 (negative is normal for GRPO)
- rewards: 0.3-0.7 range with variance
- speed: ~90s/step
- correct: trending upward over time

## Do NOT

- Kill the training unless clearly broken
- Modify llm_evaluator.py (would invalidate cache)
- Start competing GPU processes
```
