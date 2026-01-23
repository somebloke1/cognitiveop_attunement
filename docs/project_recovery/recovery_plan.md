# Cognitive Organism: Strategic Recovery Plan

**Date**: 2026-01-22  
**Status**: Ready for execution  
**Goal**: Complete original objectives (n,n+1 parallelism + student isolation) and resume training

---

## Executive Summary

The project is **healthy and ready to advance**. Core training works. Two sophisticated features exist but need integration:

1. **Student Isolation (V2 format)**: Implemented, data generated, not yet used for training
2. **(n,n+1) Parallelism**: Working implementation exists, ~1.3x speedup achieved

**This is NOT a recovery from failure** - it's integration of components built during rapid exploration.

**Critical Path**: V2 training → Evaluation → Extended training → Coverage improvement

---

## Current State Assessment

### ✅ What Works

| Component | Status | Evidence |
|-----------|--------|----------|
| **Base training** | Production-ready | 20-step test run successful |
| **GRPO pipeline** | Functional | TRL integration working |
| **Column preservation** | Fixed | `CognitiveGRPOTrainer` solves TRL bug |
| **Reward function** | Correct | Multiplicative scoring (form gates substance) |
| **Test suite** | Comprehensive | 347 tests passing |
| **V2 generation** | Ready | 10MB dataset (judgment_v2_anthropic_20260122_051114.jsonl) |
| **Async infrastructure** | Complete | `async_reward.py`, `PipelinedGRPOTrainer` |
| **Philosophical grounding** | Maintained | Documentation intact |

### ⚠️ What Needs Integration

| Component | Status | Blocker |
|-----------|--------|---------|
| **V2 training** | Not tested | Need to run training with V2 data |
| **Enhanced reward** | Not wired | Still using simple reward function |
| **Coverage analysis** | Not run | Need to analyze V2 dataset |
| **Distractor generation** | Not used | 0% of training data has distractors |
| **Async training** | Not validated | `PipelinedGRPOTrainer` untested in production |

### 📊 Data Inventory

**V1 Format** (conditions given to student - data leakage):
- `judgment_train.jsonl`: 405 examples
- `judgment_val.jsonl`: 45 examples
- Coverage: 23.3% (35/150 cells)

**V2 Format** (student isolation - correct architecture):
- `judgment_v2_anthropic_20260122_051114.jsonl`: ~500 examples (10MB)
- Student sees: proposition + evidence ONLY
- Oracle answer key: conditions, temporal context (for evaluator)
- Coverage: Unknown (needs analysis)

---

## The Two Original Goals

### Goal 1: Student Isolation (V2 Format)

**Problem**: V1 format gave conditions to student → pattern matching, not derivation

**Solution**: V2 dual-packet architecture
- **Student packet**: proposition + evidence ONLY
- **Evaluator packet**: + oracle conditions, temporal context, expected judgment
- **Forces**: Genuine condition derivation, not memorization

**Status**: ✅ Generation ready, ⚠️ training not tested

**Why it matters**: 
- Pattern matching fails on novel cases
- Intelligent inquiry generalizes
- This is the difference between surface imitation and genuine operation

### Goal 2: (n,n+1) Parallelism

**Vision**: Hide Gemini latency by overlapping with generation

**Achieved**: ~1.3x speedup (not theoretical 6.8x, but valuable)

**Implementations**:
1. `PipelinedGRPOTrainer` (434 lines) - Within-step overlap, keeps TRL features
2. `PipelinedTrainer` (807 lines) - Cross-step overlap, loses TRL features

**Status**: ✅ Working, ⚠️ not validated in production

**Why speedup is 1.3x not 6.8x**: Generation time (59-178s) often exceeds Gemini time (~58s)

---

## Strategic Priorities

### Phase 1: Validate V2 Architecture (IMMEDIATE)

**Objective**: Prove V2 format improves generalization over V1

**Success Criteria**:
- [ ] V2 training completes without errors
- [ ] Model produces properly structured judgments
- [ ] Condition derivation quality measurable
- [ ] A/B comparison shows V2 > V1 on held-out test

**Action Items**:

1. **Analyze V2 dataset coverage** (1 hour)
   ```bash
   python scripts/analyze_coverage.py \
     --data data/oracle_generated/judgment_v2_anthropic_20260122_051114.jsonl \
     --output docs/v2_coverage_analysis.md
   ```
   - Identify: domain distribution, difficulty balance, judgment type ratios
   - Compare to V1 coverage (23.3%)
   - Document gaps

2. **Split V2 dataset** (15 minutes)
   ```bash
   python -c "
   import json
   from pathlib import Path
   
   data = [json.loads(line) for line in 
           Path('data/oracle_generated/judgment_v2_anthropic_20260122_051114.jsonl').read_text().splitlines()]
   
   # 90/10 split
   split = int(len(data) * 0.9)
   train = data[:split]
   val = data[split:]
   
   Path('data/oracle_generated/judgment_v2_train.jsonl').write_text(
       '\n'.join(json.dumps(ex) for ex in train) + '\n'
   )
   Path('data/oracle_generated/judgment_v2_val.jsonl').write_text(
       '\n'.join(json.dumps(ex) for ex in val) + '\n'
   )
   print(f'Split: {len(train)} train, {len(val)} val')
   "
   ```

3. **Test V2 training (20-step sanity check)** (30 minutes)
   ```bash
   source .venv/bin/activate
   CUDA_VISIBLE_DEVICES=0 python scripts/run_enhanced_training.py \
     --data data/oracle_generated/judgment_v2_train.jsonl \
     --val-data data/oracle_generated/judgment_v2_val.jsonl \
     --output models/judgment_v2_test \
     --steps 20 \
     --batch-size 1 \
     --num-generations 5 \
     --clean
   ```
   - Monitor: loss curve, reward variance, gradient norms
   - Verify: no column dropping (reward_std > 0)
   - Check: model outputs match expected format

4. **Evaluate V2 adapter** (1 hour)
   - Load trained adapter
   - Test on 20 held-out examples
   - Manually inspect:
     * Does it derive conditions? (not just copy from input)
     * Does temporal analysis show understanding?
     * Are judgments correct?
   - Document findings

5. **A/B comparison: V1 vs V2** (2 hours)
   - Create 50-example held-out test set (not in either training set)
   - Evaluate both adapters on same test set
   - Metrics:
     * Judgment accuracy (Yes/No/Insufficient)
     * Condition derivation quality (human eval)
     * Temporal analysis coherence (human eval)
   - Decision criterion: V2 must show improvement on condition derivation

**Estimated Effort**: 5 hours  
**Risk**: Low (infrastructure exists, just needs execution)  
**Value**: HIGH - validates core architectural decision

---

### Phase 2: Extended V2 Training (SHORT-TERM)

**Objective**: Train production-quality V2 adapter

**Prerequisites**: Phase 1 complete, V2 validated as superior

**Success Criteria**:
- [ ] 100+ step training completes
- [ ] Loss converges (not oscillating)
- [ ] Validation performance stable
- [ ] Model generalizes to novel domains

**Action Items**:

1. **Decide on trainer** (30 minutes)
   
   **Recommendation**: Use `CognitiveGRPOTrainer` (non-async) for first production run
   
   **Rationale**:
   - ✅ Production-ready and tested
   - ✅ Full TRL features (checkpointing, eval, logging)
   - ✅ Easy to maintain
   - ❌ No speedup (baseline performance)
   
   **Alternative**: `PipelinedGRPOTrainer` (async) for 1.2x speedup
   - ⚠️ Not tested in production
   - ✅ Keeps TRL features
   - ⚠️ Adds complexity
   
   **Decision**: Start with `CognitiveGRPOTrainer`. If training time is prohibitive, switch to `PipelinedGRPOTrainer`.

2. **Run extended training** (4-8 hours wall time)
   ```bash
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
   ```
   - Monitor: TensorBoard logs, loss curves
   - Checkpoints: Every 25 steps (for recovery)
   - Eval: Every 25 steps (catch overfitting early)

3. **Evaluate on held-out test set** (2 hours)
   - Use 100-example test set (not in training or validation)
   - Metrics:
     * Judgment accuracy by domain
     * Judgment accuracy by difficulty
     * Condition derivation quality (sample 20, human eval)
     * Temporal analysis quality (sample 20, human eval)
   - Compare to baseline (untrained model)

4. **Error analysis** (2 hours)
   - Identify failure modes:
     * Which domains have lowest accuracy?
     * Which difficulty levels fail?
     * Common patterns in wrong judgments?
   - Document for next iteration

**Estimated Effort**: 8-12 hours  
**Risk**: Medium (training may reveal issues)  
**Value**: HIGH - production adapter for judgment operation

---

### Phase 3: Coverage Improvement (MEDIUM-TERM)

**Objective**: Fill coverage gaps to >80% of 150 cells

**Prerequisites**: Phase 2 complete, error analysis done

**Success Criteria**:
- [ ] Coverage >80% (120/150 cells)
- [ ] All domains represented
- [ ] All difficulty levels represented
- [ ] 30% of data has distractors

**Current Coverage**: 23.3% (V1), unknown (V2)

**Coverage Matrix**: 35 domains × 3 judgments × 5 difficulties × 2 distractor states = 1050 cells (simplified to 150 by grouping)

**Action Items**:

1. **Analyze V2 coverage** (1 hour)
   - Run coverage analyzer on V2 dataset
   - Identify gaps:
     * Missing domains
     * Missing difficulty levels
     * Missing judgment types
     * Zero distractors
   - Prioritize by error analysis (which gaps hurt most?)

2. **Generate targeted examples** (4-8 hours)
   - Use `generate_with_model_v2.py` with filters:
     ```bash
     # Example: Fill gaps in formal domains, expert difficulty
     python scripts/generate_with_model_v2.py \
       --model gemini/gemini-2.5-pro \
       --mode formal \
       --difficulty expert \
       --count 50 \
       --output data/oracle_generated/judgment_v2_formal_expert.jsonl
     ```
   - Target: 200-300 additional examples
   - Focus: Identified gaps from coverage analysis

3. **Generate distractors** (2-4 hours)
   - Use `distractor_generator.py` (implemented, not wired)
   - Target: 30% of training data
   - Types:
     * `misaligned_phase`: Wrong cognitive level language
     * `surface_match`: Superficial pattern without substance
     * `temporal_confusion`: Aspectual errors
   - Integrate into training data

4. **Combine and re-split** (30 minutes)
   ```bash
   # Combine V2 base + targeted + distractors
   cat data/oracle_generated/judgment_v2_*.jsonl > data/oracle_generated/judgment_v2_combined.jsonl
   
   # Re-split 90/10
   python scripts/split_dataset.py \
     --input data/oracle_generated/judgment_v2_combined.jsonl \
     --train data/oracle_generated/judgment_v2_train_full.jsonl \
     --val data/oracle_generated/judgment_v2_val_full.jsonl \
     --ratio 0.9
   ```

5. **Verify coverage** (30 minutes)
   ```bash
   python scripts/analyze_coverage.py \
     --data data/oracle_generated/judgment_v2_train_full.jsonl \
     --output docs/v2_coverage_full.md
   ```
   - Confirm: >80% coverage
   - Document: remaining gaps (acceptable if rare edge cases)

**Estimated Effort**: 8-14 hours  
**Risk**: Low (generation pipeline proven)  
**Value**: MEDIUM - improves robustness, not core functionality

---

### Phase 4: Enhanced Reward Integration (MEDIUM-TERM)

**Objective**: Wire sophisticated reward function into training

**Prerequisites**: Phase 2 complete (baseline V2 training working)

**Success Criteria**:
- [ ] Enhanced reward function integrated
- [ ] Training converges with enhanced reward
- [ ] Evidence grounding bonus improves citation quality
- [ ] Domain verification bonus improves domain-specific reasoning

**Current State**: `enhanced_reward.py` implemented (54 tests passing), not used

**Action Items**:

1. **Review enhanced reward implementation** (1 hour)
   - Read: `src/training/enhanced_reward.py`
   - Understand: Scoring components
     * Base correctness: +0.6/-0.5 (multiplicative gate)
     * Evidence grounding: +0.2 for citations
     * Domain verification: +0.1 for domain-specific reasoning
     * Structure: +0.1 for format compliance
   - Verify: Tests pass (should be 54/54)

2. **Wire into training script** (2 hours)
   - Modify: `scripts/run_enhanced_training.py`
   - Add flag: `--use-enhanced-reward`
   - Integration:
     ```python
     if args.use_enhanced_reward:
         from src.training.enhanced_reward import EnhancedRewardFunction
         reward_fn = EnhancedRewardFunction(
             llm_evaluator=llm_evaluator,
             evidence_weight=0.2,
             domain_weight=0.1,
             structure_weight=0.1,
         )
     else:
         reward_fn = hybrid_reward_fn  # Current simple reward
     ```
   - Test: 20-step run with enhanced reward

3. **A/B comparison: Simple vs Enhanced reward** (4 hours)
   - Train two adapters (same data, different rewards)
   - Compare:
     * Convergence speed (does enhanced reward help?)
     * Final performance (accuracy on test set)
     * Citation quality (evidence grounding working?)
     * Domain reasoning (domain verification working?)
   - Decision: Keep enhanced if improvement >5%

4. **Tune reward weights** (2-4 hours)
   - If enhanced reward helps, optimize weights
   - Grid search or manual tuning:
     * Evidence weight: 0.1, 0.2, 0.3
     * Domain weight: 0.05, 0.1, 0.15
     * Structure weight: 0.05, 0.1, 0.15
   - Criterion: Validation performance

**Estimated Effort**: 8-12 hours  
**Risk**: Medium (reward tuning can be finicky)  
**Value**: MEDIUM - incremental improvement, not core functionality

---

### Phase 5: Async Training Validation (OPTIONAL)

**Objective**: Validate `PipelinedGRPOTrainer` for 1.2-1.3x speedup

**Prerequisites**: Phase 2 complete (baseline training working)

**Success Criteria**:
- [ ] Async training completes without errors
- [ ] Speedup measured: >1.2x
- [ ] Final performance matches non-async (no quality degradation)

**Action Items**:

1. **Test async training (20-step)** (1 hour)
   ```bash
   source .venv/bin/activate
   CUDA_VISIBLE_DEVICES=0 python scripts/run_enhanced_training.py \
     --data data/oracle_generated/judgment_v2_train.jsonl \
     --val-data data/oracle_generated/judgment_v2_val.jsonl \
     --output models/judgment_v2_async_test \
     --steps 20 \
     --batch-size 1 \
     --num-generations 5 \
     --async \
     --clean
   ```
   - Monitor: Timing logs (wait times should be ~0s)
   - Verify: Reward variance > 0 (no column dropping)
   - Check: Loss curve similar to non-async

2. **Measure speedup** (30 minutes)
   - Compare: Async vs non-async on same 20-step run
   - Metric: Total wall time
   - Expected: 1.2-1.3x speedup
   - Document: Actual speedup achieved

3. **Extended async training** (4-8 hours)
   - If 20-step test succeeds, run 100-step training
   - Compare final performance to non-async
   - Decision: Use async if speedup >1.2x AND no quality degradation

**Estimated Effort**: 6-10 hours  
**Risk**: Low (infrastructure exists, just needs validation)  
**Value**: LOW - nice-to-have, not essential

---

## Dependency Graph

```
Phase 1: Validate V2
    ↓
Phase 2: Extended V2 Training ←─────┐
    ↓                               │
Phase 3: Coverage Improvement ──────┘
    ↓
Phase 4: Enhanced Reward (optional, can run in parallel with Phase 3)
    ↓
Phase 5: Async Training (optional, can run in parallel with Phase 4)
```

**Critical Path**: Phase 1 → Phase 2 → Phase 3

**Optional Enhancements**: Phase 4, Phase 5 (can be deferred)

---

## Risk Assessment

### Low Risk ✅

- **V2 generation**: Already working, 10MB dataset exists
- **Coverage analysis**: Tool exists, just needs to run
- **Distractor generation**: Implemented, tested (67 tests passing)
- **Async infrastructure**: Complete, tested in isolation

### Medium Risk ⚠️

- **V2 training**: Format change may reveal issues
  - **Mitigation**: 20-step sanity check before extended training
- **Enhanced reward**: Reward tuning can be finicky
  - **Mitigation**: A/B test, keep simple reward as fallback
- **Coverage gaps**: May be hard to fill some cells
  - **Mitigation**: Accept 80% coverage, not 100%

### High Risk ❌

- **None identified**

---

## Success Metrics

### Phase 1 Success (V2 Validation)

- [ ] V2 training completes without errors
- [ ] Model derives conditions (not just copies)
- [ ] A/B test shows V2 > V1 on condition derivation

### Phase 2 Success (Extended Training)

- [ ] 100-step training converges
- [ ] Judgment accuracy >70% on held-out test
- [ ] Model generalizes across domains

### Phase 3 Success (Coverage)

- [ ] Coverage >80% (120/150 cells)
- [ ] 30% of data has distractors
- [ ] Error analysis shows balanced performance

### Overall Project Success

- [ ] Production V2 adapter trained and evaluated
- [ ] Student isolation validated as superior to V1
- [ ] Coverage gaps identified and prioritized
- [ ] Path forward clear for next iteration

---

## Timeline Estimate

| Phase | Effort | Duration (if focused) |
|-------|--------|----------------------|
| Phase 1: V2 Validation | 5 hours | 1 day |
| Phase 2: Extended Training | 8-12 hours | 1-2 days |
| Phase 3: Coverage | 8-14 hours | 1-2 days |
| Phase 4: Enhanced Reward | 8-12 hours | 1-2 days (optional) |
| Phase 5: Async Validation | 6-10 hours | 1 day (optional) |
| **Total (critical path)** | **21-31 hours** | **3-5 days** |
| **Total (with optional)** | **35-53 hours** | **5-8 days** |

**Assumptions**:
- Single developer, focused work
- No major blockers or bugs
- GPU available (RTX 3090)
- API access (Gemini for evaluation)

---

## What NOT To Do

### ❌ Don't Rewrite Working Code

- `CognitiveGRPOTrainer` works → use it
- `async_reward.py` works → use it
- Test suite passes → trust it

### ❌ Don't Pursue Theoretical Perfection

- 80% coverage is sufficient (not 100%)
- 1.2x speedup is valuable (not 6.8x)
- Simple reward may be good enough (enhanced is optional)

### ❌ Don't Lose Philosophical Grounding

- Student isolation is about GENUINE OPERATION, not just data format
- Judgment is grasping the virtually unconditioned, not pattern matching
- Form without substance is empty form

### ❌ Don't Parallelize Prematurely

- Finish Phase 1 before Phase 2
- Validate V2 before extended training
- Prove value before optimization

---

## Open Questions (To Be Resolved)

1. **V2 coverage**: What % of 150 cells are filled? (Answer in Phase 1)
2. **V2 performance**: Does student isolation improve generalization? (Answer in Phase 1)
3. **Optimal trainer**: Async or non-async for production? (Answer in Phase 2/5)
4. **Enhanced reward value**: Does it improve performance >5%? (Answer in Phase 4)
5. **Coverage target**: Is 80% sufficient or aim higher? (Answer in Phase 3)

---

## Next Session Action Plan

**Immediate (First 30 minutes)**:

1. Read this recovery plan
2. Confirm V2 dataset exists and is valid
3. Run coverage analysis on V2 data
4. Review coverage report

**First Work Block (2-3 hours)**:

1. Split V2 dataset (train/val)
2. Run 20-step V2 training test
3. Monitor logs, verify no errors
4. Inspect model outputs

**Decision Point**:

- ✅ If V2 test succeeds → Proceed to Phase 2 (extended training)
- ⚠️ If V2 test fails → Debug, document issues, revise plan

---

## Philosophical Grounding

### Why This Plan Serves the Vision

**The Vision**: Train AI systems that perform genuine cognitive operations, not surface pattern matching.

**How This Plan Serves It**:

1. **Student Isolation (V2)**: Forces condition derivation, not memorization
   - Pattern matching: "I've seen this before, copy the answer"
   - Intelligent inquiry: "What must be true for this to hold?"

2. **Coverage Improvement**: Ensures generalization across domains
   - Clustering: Model learns narrow patterns
   - Balance: Model learns general operations

3. **Enhanced Reward**: Rewards substance, not just form
   - Form without substance: "JUDGMENT: Yes" (but wrong)
   - Substance: Correct judgment with evidence-grounded reasoning

4. **Async Training**: Efficiency enables more experimentation
   - Slow training: Limited iterations, less learning
   - Fast training: More experiments, faster convergence to truth

**The Virtually Unconditioned in This Plan**:

- **Conditioned**: "V2 format improves generalization"
- **Link**: If student isolation forces derivation, then generalization improves
- **Fulfillment**: A/B test on held-out data (Phase 1)

We will grasp the virtually unconditioned through empirical verification, not assumption.

---

## Conclusion

**The project is healthy.** Core training works. Infrastructure exists. The "destabilization" was architectural exploration, not catastrophic failure.

**The path forward is clear**:

1. Validate V2 architecture (5 hours)
2. Train production V2 adapter (8-12 hours)
3. Improve coverage (8-14 hours)
4. Optionally enhance reward and async training

**Total effort**: 21-31 hours for critical path, 35-53 hours with optional enhancements.

**The vision remains intact**: Train AI systems that perform genuine cognitive operations, not surface pattern matching.

**Next action**: Execute Phase 1 (V2 Validation).

---

**Document Status**: Complete and ready for execution  
**Author**: Claude Code (project-planner agent)  
**Date**: 2026-01-22  
**Version**: 1.0
