# Project Timeline: Visual History

**Date**: 2026-01-22

---

## Timeline Overview

```
2026-01-19 (Session 1: Foundation)
    │
    ├─ Initial commit (c7f2981)
    │  ├─ Base training infrastructure (SFT, GRPO)
    │  ├─ DSPy oracle pipeline
    │  ├─ 450 verified examples generated
    │  ├─ Philosophical grounding docs
    │  └─ First adapter trained (20 steps) ✅
    │
    └─ End of Session 1: Working baseline established

2026-01-20 (Session 2: Reward Function Crisis)
    │
    ├─ Problem identified: Structure-only reward
    │  └─ Wrong judgments could score positively
    │
    ├─ Solution implemented: Correctness-dominant reward
    │  ├─ Multiplicative gating
    │  ├─ Ground truth passed via TRL kwargs
    │  └─ "Form without substance = empty form"
    │
    └─ End of Session 2: Reward function fixed

2026-01-20 (Session 3: Training Collapse & Recovery)
    │
    ├─ Training collapse observed
    │  ├─ loss = 0.0
    │  ├─ reward_std = 0.0
    │  ├─ grad_norm = 0.0
    │  └─ No learning signal
    │
    ├─ Root cause: TRL drops custom columns
    │  └─ _set_signature_columns_if_needed() hardcodes defaults
    │
    ├─ Fix: CognitiveGRPOTrainer subclass
    │  └─ Preserves expected_judgment, domain, etc.
    │
    ├─ Test architecture expansion (28e761e)
    │  ├─ 5 modules implemented (TDD approach)
    │  ├─ 331 tests passing
    │  └─ NOT YET WIRED into training
    │
    └─ End of Session 3: Training fixed, modules ready

2026-01-21 (Session 4: Refinement)
    │
    ├─ Domain API refactoring
    │  ├─ Unified principled schema
    │  ├─ Removed legacy parallel system
    │  └─ All 329 tests still passing
    │
    ├─ Student isolation architecture
    │  ├─ V2 format designed (dual-packet)
    │  ├─ generate_with_model_v2.py created
    │  ├─ postprocess_training_data.py created
    │  └─ NOT YET USED for training
    │
    ├─ Temporal structure integration
    │  ├─ Philosophical dialogue (tmp/ex1.md)
    │  ├─ Temporal focus types implemented
    │  └─ Pervading dimension (not separate type)
    │
    └─ End of Session 4: Architecture refined

2026-01-22 (Session 5: Pipelined Training)
    │
    ├─ Pipelined training tested
    │  ├─ pipelined_trainer.py (807 lines)
    │  ├─ pipelined_grpo_trainer.py (434 lines)
    │  ├─ async_reward.py (749 lines)
    │  └─ Within-step overlap working (~1.3x speedup)
    │
    ├─ Logs show success
    │  ├─ Step 0: 120s (synchronous)
    │  ├─ Step 2+: 85-95s (wait=0.0s)
    │  └─ Gemini ready immediately (overlap working!)
    │
    └─ Current state: Multiple implementations, integration needed

2026-01-22 (This Analysis)
    │
    └─ Project recovery documentation
       ├─ history.md (comprehensive analysis)
       ├─ executive_summary.md (quick reference)
       └─ timeline.md (this document)
```

---

## Parallel Development Tracks

### Track 1: Core Training (Continuous)

```
Session 1 ─────────────────────────────────────────────────────────────>
    │                                                                    │
    ├─ SFT trainer                                                      │
    ├─ GRPO trainer (TRL)                                               │
    ├─ Data generation                                                  │
    ├─ Oracle verification                                              │
    └─ First adapter trained ✅                                         │
                                                                         │
Session 2 ─────────────────────────────────────────────────────────────>
    │                                                                    │
    ├─ Reward function fixed                                            │
    └─ Ground truth passed ✅                                           │
                                                                         │
Session 3 ─────────────────────────────────────────────────────────────>
    │                                                                    │
    ├─ TRL bug discovered                                               │
    ├─ CognitiveGRPOTrainer created                                     │
    └─ Training working again ✅                                        │
                                                                         │
Session 5 ─────────────────────────────────────────────────────────────>
    │                                                                    │
    └─ Pipelined training tested ✅                                     │
                                                                         │
                                                                    PRESENT
```

### Track 2: Test Architecture (Burst)

```
Session 3 ─────────────────────────────────────────────────────────────>
    │                                                                    │
    ├─ Research external framework                                      │
    ├─ Extract 10 insights                                              │
    ├─ Critical reflection (6 YES, 4 INSUFFICIENT)                      │
    ├─ Implementation plan (1463 lines)                                 │
    ├─ TDD: 331 tests written                                           │
    ├─ 5 modules implemented                                            │
    └─ All tests passing ✅                                             │
                                                                         │
                                                                    PRESENT
                                                                         │
                                                                         ▼
                                                            NOT YET WIRED ⚠️
```

### Track 3: Parallelism (Exploration)

```
Session 1 ─────────────────────────────────────────────────────────────>
    │                                                                    │
    └─ Idea: Overlap Gemini with generation                             │
                                                                         │
Session 3 ─────────────────────────────────────────────────────────────>
    │                                                                    │
    ├─ async_reward.py created (749 lines)                              │
    ├─ AsyncGeminiEvaluator (low-level)                                 │
    ├─ PipelinedHybridReward (high-level)                               │
    └─ Infrastructure complete ✅                                       │
                                                                         │
Session 4 ─────────────────────────────────────────────────────────────>
    │                                                                    │
    ├─ pipelined_trainer.py (807 lines)                                 │
    │  ├─ Custom training loop                                          │
    │  ├─ Explicit state machine                                        │
    │  └─ Full control, loses TRL features                              │
    │                                                                    │
    ├─ pipelined_grpo_trainer.py (434 lines)                            │
    │  ├─ TRL subclass                                                  │
    │  ├─ Within-step overlap works                                     │
    │  └─ Cross-step overlap attempted                                  │
    │                                                                    │
    └─ run_pipelined_training.py (script)                               │
                                                                         │
Session 5 ─────────────────────────────────────────────────────────────>
    │                                                                    │
    ├─ Pipelined training tested                                        │
    ├─ Logs show overlap working                                        │
    ├─ ~1.3x speedup achieved                                           │
    └─ Within-step overlap sufficient ✅                                │
                                                                         │
                                                                    PRESENT
                                                                         │
                                                                         ▼
                                                         PARTIAL SUCCESS ⚠️
                                                    (within-step works,
                                                     cross-step attempted)
```

### Track 4: Student Isolation (Design)

```
Session 1 ─────────────────────────────────────────────────────────────>
    │                                                                    │
    └─ V1 format: Conditions in user message                            │
                                                                         │
Session 2 ─────────────────────────────────────────────────────────────>
    │                                                                    │
    └─ Concern: Data leakage identified                                 │
                                                                         │
Session 4 ─────────────────────────────────────────────────────────────>
    │                                                                    │
    ├─ V2 format designed (dual-packet)                                 │
    ├─ generate_with_model_v2.py (1479 lines)                           │
    ├─ postprocess_training_data.py (579 lines)                         │
    ├─ Student sees: proposition + evidence only                        │
    ├─ Evaluator sees: + oracle answer key                              │
    └─ Scripts complete ✅                                              │
                                                                         │
                                                                    PRESENT
                                                                         │
                                                                         ▼
                                                            NOT YET USED ⚠️
                                                    (current data still V1)
```

---

## Decision Points

### Decision 1: Reward Function (Session 2)

```
Problem: Structure-only reward
    │
    ├─ Option A: Add correctness check (simple)
    ├─ Option B: Multiplicative gating (sophisticated)
    └─ Option C: Separate correctness/structure rewards
    
Decision: Option B (multiplicative gating)
    │
    └─ Rationale: "Form without substance = empty form"
```

### Decision 2: TRL Bug Fix (Session 3)

```
Problem: Custom columns dropped
    │
    ├─ Option A: Fork TRL and modify
    ├─ Option B: Subclass and override
    └─ Option C: Abandon TRL, custom trainer
    
Decision: Option B (subclass)
    │
    └─ Rationale: Minimal change, keeps TRL features
```

### Decision 3: Pipelined Architecture (Sessions 3-5)

```
Problem: Gemini latency bottleneck
    │
    ├─ Option A: Custom training loop (full control)
    ├─ Option B: TRL subclass (keep features)
    └─ Option C: Hybrid (custom + TRL components)
    
Decision: Explored A and B in parallel
    │
    ├─ A: pipelined_trainer.py (complete)
    ├─ B: pipelined_grpo_trainer.py (partial)
    └─ Result: Both exist, B recommended for production
```

### Decision 4: Student Isolation (Session 4)

```
Problem: Data leakage (conditions given)
    │
    ├─ Option A: Remove conditions from user message
    ├─ Option B: Dual-packet (student vs evaluator)
    └─ Option C: Gradual reveal (curriculum)
    
Decision: Option B (dual-packet)
    │
    └─ Rationale: Clean separation, evaluator has answer key
```

---

## Integration Status Matrix

| Component | Implemented | Tested | Wired | Used |
|-----------|-------------|--------|-------|------|
| **Core Training** |
| SFT trainer | ✅ | ✅ | ✅ | ✅ |
| GRPO trainer (TRL) | ✅ | ✅ | ✅ | ✅ |
| CognitiveGRPOTrainer | ✅ | ✅ | ✅ | ✅ |
| Correctness reward | ✅ | ✅ | ✅ | ✅ |
| **Parallelism** |
| async_reward.py | ✅ | ✅ | ✅ | ✅ |
| pipelined_trainer.py | ✅ | ✅ | ⚠️ | ❌ |
| pipelined_grpo_trainer.py | ✅ | ✅ | ⚠️ | ❌ |
| run_pipelined_training.py | ✅ | ✅ | ⚠️ | ❌ |
| **Test Architecture** |
| extended_schema.py | ✅ | ✅ | ❌ | ❌ |
| coverage/analyzer.py | ✅ | ✅ | ❌ | ❌ |
| evidence_grounding.py | ✅ | ✅ | ❌ | ❌ |
| distractor_generator.py | ✅ | ✅ | ❌ | ❌ |
| enhanced_reward.py | ✅ | ✅ | ❌ | ❌ |
| **Student Isolation** |
| generate_with_model_v2.py | ✅ | ✅ | ❌ | ❌ |
| postprocess_training_data.py | ✅ | ✅ | ❌ | ❌ |
| V2 dataset | ❌ | ❌ | ❌ | ❌ |

**Legend**:
- ✅ Complete
- ⚠️ Partial
- ❌ Not done

---

## Architectural Evolution

### Phase 1: Simple Sequential (Session 1)

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Loop                            │
│                                                             │
│  for step in steps:                                         │
│      ┌──────────────┐                                       │
│      │  Generate    │  ~5s (GPU)                            │
│      └──────┬───────┘                                       │
│             │                                               │
│             ▼                                               │
│      ┌──────────────┐                                       │
│      │  Reward Fn   │  ~60s (Gemini blocking)               │
│      └──────┬───────┘                                       │
│             │                                               │
│             ▼                                               │
│      ┌──────────────┐                                       │
│      │  Update      │  ~3s (GPU)                            │
│      └──────────────┘                                       │
│                                                             │
│  Total: ~68s per step                                       │
└─────────────────────────────────────────────────────────────┘
```

### Phase 2: Within-Step Overlap (Session 5)

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Loop                            │
│                                                             │
│  for step in steps:                                         │
│      ┌──────────────┐                                       │
│      │  Generate    │  ~5s (GPU)                            │
│      └──────┬───────┘                                       │
│             │                                               │
│             ▼                                               │
│      ┌──────────────┐                                       │
│      │ Fire Gemini  │  ~0s (async, returns immediately)     │
│      └──────┬───────┘                                       │
│             │                                               │
│             ├─────────────────────────────┐                 │
│             │                             │                 │
│             ▼                             ▼                 │
│      ┌──────────────┐              ┌──────────────┐         │
│      │ Compute      │  ~3s (GPU)   │   Gemini     │ ~58s    │
│      │ Log Probs    │              │   (async)    │         │
│      └──────┬───────┘              └──────┬───────┘         │
│             │                             │                 │
│             └─────────────┬───────────────┘                 │
│                           │                                 │
│                           ▼                                 │
│                    ┌──────────────┐                         │
│                    │ Wait Gemini  │  ~0s (if ready)         │
│                    └──────┬───────┘                         │
│                           │                                 │
│                           ▼                                 │
│                    ┌──────────────┐                         │
│                    │  Update      │  ~3s (GPU)              │
│                    └──────────────┘                         │
│                                                             │
│  Total: max(8s, 58s) ≈ 58s per step                         │
│  Speedup: ~1.2x (modest but real)                           │
└─────────────────────────────────────────────────────────────┘
```

### Phase 3: Cross-Step Overlap (Attempted, Not Achieved)

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Loop                            │
│                                                             │
│  Step 0:                                                    │
│      Generate(0) → Fire_Gemini(0) → Wait(0) → Update(0)    │
│                                                             │
│  Step 1:                                                    │
│      Generate(1) → Fire_Gemini(1) → [pending=1]            │
│                                                             │
│  Step N (N > 1):                                            │
│      ┌──────────────┐                                       │
│      │ Generate(N)  │  ~5s (GPU)                            │
│      └──────┬───────┘                                       │
│             │                                               │
│             ▼                                               │
│      ┌──────────────┐                                       │
│      │Fire_Gemini(N)│  ~0s (async)                          │
│      └──────┬───────┘                                       │
│             │                                               │
│             │  ┌─────────────────────────────────┐          │
│             │  │  Gemini(N-1) was running        │          │
│             │  │  during Generate(N)             │          │
│             │  │  → Latency hidden!              │          │
│             │  └─────────────────────────────────┘          │
│             │                                               │
│             ▼                                               │
│      ┌──────────────┐                                       │
│      │Wait_Gemini   │  ~0s (should be ready)                │
│      │   (N-1)      │                                       │
│      └──────┬───────┘                                       │
│             │                                               │
│             ▼                                               │
│      ┌──────────────┐                                       │
│      │ Update(N-1)  │  ~3s (GPU)                            │
│      └──────┬───────┘                                       │
│             │                                               │
│             └─ [pending=N]                                  │
│                                                             │
│  Final:                                                     │
│      Wait_Gemini(last) → Update(last)                       │
│                                                             │
│  Total: max(8s, gemini_latency) ≈ 10s per step             │
│  Speedup: ~6.8x (theoretical)                               │
│                                                             │
│  Status: Attempted in pipelined_grpo_trainer.py             │
│          TRL's caching makes this difficult                 │
│          Custom loop (pipelined_trainer.py) achieves this   │
└─────────────────────────────────────────────────────────────┘
```

---

## Lessons Learned

### Technical Lessons

1. **TRL is opinionated**
   - Hardcodes column names
   - Caches generation results
   - Requires subclassing to customize

2. **Async is tricky**
   - Thread safety matters
   - Event loops need careful management
   - Within-step easier than cross-step

3. **Speedup depends on bottleneck**
   - If gen > gemini: overlap hides all latency
   - If gen < gemini: still wait
   - Actual speedup: ~1.3x (not theoretical 6.8x)

4. **TDD works**
   - 331 tests written before implementation
   - All modules work in isolation
   - Integration is separate concern

### Philosophical Lessons

5. **Form without substance is empty**
   - Structure without correctness = no learning
   - Multiplicative gating enforces this

6. **Data leakage is subtle**
   - Giving conditions ≠ deriving conditions
   - Pattern matching ≠ intelligent inquiry
   - Student isolation forces genuine operation

7. **Operations are recurrent**
   - Not sequential pipeline
   - Judgment → Understanding → Attention
   - Current isolation is temporary scaffold

### Process Lessons

8. **Multiple implementations = exploration**
   - Not failure, but search
   - Each addresses different tradeoffs
   - Convergence happens later

9. **Documentation prevents decay**
   - Philosophical grounding maintained
   - Succession notes detailed
   - Future agents can recover

10. **Context compaction is real**
    - Rapid iteration visible in timestamps
    - Multiple versions of scripts
    - Commented-out code (attempted approaches)

---

## Next Steps (Prioritized)

### P0: Integration (Immediate)

1. Choose `pipelined_grpo_trainer.py` for production
2. Generate V2 dataset (500 examples, student isolation)
3. Test `run_enhanced_training.py` orchestration
4. A/B test: V1 vs V2 format

### P1: Optimization (Short-term)

5. Profile pipelined training (where is time spent?)
6. Experiment with faster Gemini model
7. Integrate `enhanced_reward.py`
8. Fill coverage gaps (23.3% → >80%)

### P2: Validation (Medium-term)

9. Extended training run (500-1000 steps)
10. Human evaluation (100-200 labeled examples)
11. Adversarial testing (novel domains, edge cases)
12. Measure generalization (held-out test set)

### P3: Expansion (Long-term)

13. Train other cognitive levels (Attention, Understanding, Decision)
14. Network architecture (feedback loops)
15. Self-appropriation (meta-cognitive prompts)
16. Production deployment (API, use cases)

---

**For Full Details**: See `history.md` (comprehensive analysis)  
**For Quick Reference**: See `executive_summary.md`