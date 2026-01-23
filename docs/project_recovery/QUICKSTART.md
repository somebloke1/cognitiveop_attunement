# Recovery Plan: Quick Start Guide

**Read this first, then see `recovery_plan.md` for full details.**

---

## TL;DR

You have:
- ✅ Working training pipeline
- ✅ V2 student isolation format (10MB dataset ready)
- ✅ Async infrastructure (1.3x speedup available)
- ⚠️ Components not yet integrated

**Next step**: Validate V2 format improves over V1, then train production adapter.

---

## Phase 1: V2 Validation (START HERE)

**Time**: 5 hours  
**Goal**: Prove V2 student isolation works

### Commands

```bash
# 1. Analyze V2 coverage (1 hour)
source .venv/bin/activate
python scripts/analyze_coverage.py \
  --data data/oracle_generated/judgment_v2_anthropic_20260122_051114.jsonl \
  --output docs/v2_coverage_analysis.md

# 2. Split dataset (5 minutes)
python -c "
import json
from pathlib import Path

data = [json.loads(line) for line in 
        Path('data/oracle_generated/judgment_v2_anthropic_20260122_051114.jsonl').read_text().splitlines()]

split = int(len(data) * 0.9)
train, val = data[:split], data[split:]

Path('data/oracle_generated/judgment_v2_train.jsonl').write_text(
    '\n'.join(json.dumps(ex) for ex in train) + '\n'
)
Path('data/oracle_generated/judgment_v2_val.jsonl').write_text(
    '\n'.join(json.dumps(ex) for ex in val) + '\n'
)
print(f'Split: {len(train)} train, {len(val)} val')
"

# 3. Test V2 training (30 minutes)
CUDA_VISIBLE_DEVICES=0 python scripts/run_enhanced_training.py \
  --data data/oracle_generated/judgment_v2_train.jsonl \
  --val-data data/oracle_generated/judgment_v2_val.jsonl \
  --output models/judgment_v2_test \
  --steps 20 \
  --batch-size 1 \
  --num-generations 5 \
  --clean

# 4. Inspect outputs (1 hour)
# Load adapter, test on examples, manually verify condition derivation

# 5. A/B test V1 vs V2 (2 hours)
# Create held-out test set, evaluate both adapters, compare
```

### Success Criteria

- [ ] V2 training completes without errors
- [ ] Model derives conditions (not just copies from input)
- [ ] V2 shows improvement over V1 on condition derivation quality

### If This Fails

- Check logs for errors
- Verify V2 format matches expected structure
- Document issues in `docs/project_recovery/v2_issues.md`
- Revise plan

---

## Phase 2: Extended Training (AFTER PHASE 1)

**Time**: 8-12 hours  
**Goal**: Train production V2 adapter

### Commands

```bash
# Extended training (4-8 hours wall time)
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python scripts/run_enhanced_training.py \
  --data data/oracle_generated/judgment_v2_train.jsonl \
  --val-data data/oracle_generated/judgment_v2_val.jsonl \
  --output models/judgment_v2_extended \
  --steps 100 \
  --batch-size 1 \
  --num-generations 5 \
  --save-steps 25 \
  --eval-steps 25 \
  --clean

# Monitor with TensorBoard
tensorboard --logdir models/judgment_v2_extended/logs
```

### Success Criteria

- [ ] 100-step training converges
- [ ] Judgment accuracy >70% on held-out test
- [ ] Model generalizes across domains

---

## Phase 3: Coverage Improvement (AFTER PHASE 2)

**Time**: 8-14 hours  
**Goal**: Fill coverage gaps to >80%

### Commands

```bash
# 1. Analyze coverage
python scripts/analyze_coverage.py \
  --data data/oracle_generated/judgment_v2_train.jsonl \
  --output docs/v2_coverage_initial.md

# 2. Generate targeted examples (adjust filters based on gaps)
python scripts/generate_with_model_v2.py \
  --model gemini/gemini-2.5-pro \
  --mode formal \
  --difficulty expert \
  --count 50 \
  --output data/oracle_generated/judgment_v2_formal_expert.jsonl

# 3. Combine datasets
cat data/oracle_generated/judgment_v2_*.jsonl > \
  data/oracle_generated/judgment_v2_combined.jsonl

# 4. Re-split
# (Use split script from Phase 1)

# 5. Verify coverage
python scripts/analyze_coverage.py \
  --data data/oracle_generated/judgment_v2_train_full.jsonl \
  --output docs/v2_coverage_full.md
```

### Success Criteria

- [ ] Coverage >80% (120/150 cells)
- [ ] All domains represented
- [ ] 30% of data has distractors

---

## Optional Enhancements

### Phase 4: Enhanced Reward (Optional)

**Time**: 8-12 hours  
**Value**: Incremental improvement

Wire `enhanced_reward.py` into training, A/B test vs simple reward.

### Phase 5: Async Training (Optional)

**Time**: 6-10 hours  
**Value**: 1.2-1.3x speedup

Validate `PipelinedGRPOTrainer` with `--async` flag.

---

## Key Files

### Training Scripts

- `scripts/run_enhanced_training.py` - **PRIMARY** training script
- `scripts/generate_with_model_v2.py` - **PRIMARY** generation script
- `scripts/analyze_coverage.py` - Coverage analysis

### Trainers

- `src/training/cognitive_grpo_trainer.py` - **RECOMMENDED** (production-ready)
- `src/training/pipelined_grpo_trainer.py` - Async variant (untested)
- `src/training/pipelined_trainer.py` - Custom loop (alternative)

### Data

- `data/oracle_generated/judgment_v2_anthropic_20260122_051114.jsonl` - **V2 dataset** (10MB)
- `data/oracle_generated/judgment_train.jsonl` - V1 dataset (405 examples)

### Documentation

- `docs/project_recovery/recovery_plan.md` - **FULL PLAN** (this is the summary)
- `docs/project_recovery/executive_summary.md` - Project state overview
- `docs/project_recovery/trainer_clarification.md` - Trainer comparison

---

## Decision Points

### Which Trainer?

**Recommendation**: `CognitiveGRPOTrainer` (non-async)

- ✅ Production-ready
- ✅ Full TRL features
- ❌ No speedup

**Alternative**: `PipelinedGRPOTrainer` (async) if training time is prohibitive

### Enhanced Reward?

**Recommendation**: Start with simple reward, add enhanced if needed

- Simple reward: Correctness-dominant multiplicative scoring (working)
- Enhanced reward: + evidence grounding + domain verification (untested)

### Coverage Target?

**Recommendation**: 80% coverage (120/150 cells)

- 100% coverage is overkill
- 80% ensures balanced representation
- Focus on quality over exhaustive coverage

---

## What NOT To Do

- ❌ Don't rewrite working code
- ❌ Don't pursue 100% coverage
- ❌ Don't optimize before validating
- ❌ Don't lose philosophical grounding

---

## Next Action

**Execute Phase 1**: Validate V2 architecture

1. Run coverage analysis
2. Split dataset
3. Test 20-step training
4. Inspect outputs
5. A/B test V1 vs V2

**Estimated time**: 5 hours

**Decision point**: If V2 validates, proceed to Phase 2. If not, debug and revise.

---

**For full details, see**: `docs/project_recovery/recovery_plan.md`
