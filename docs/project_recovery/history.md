# Project Recovery: Comprehensive Historical Analysis

**Date**: 2026-01-22  
**Analyst**: Claude Code (Project Historian Agent)  
**Purpose**: Trace the evolution of the Cognitive Organism project, with special focus on the destabilization during (n,n+1) parallelism implementation and the student isolation architecture

---

## Executive Summary

The Cognitive Organism project successfully implemented a working GRPO training pipeline for teaching Lonergan's cognitive operation of JUDGMENT to Qwen2.5-7B-Instruct. However, the project went through significant architectural churn during attempts to implement two sophisticated features:

1. **(n,n+1) Parallelism**: Overlapping Gemini cloud evaluation of step N with local model generation of step N+1 to hide 5-10s of API latency per training step
2. **Student Isolation**: Preventing data leakage by ensuring the student model derives conditions and temporal analysis rather than being given them

**Current State**: Both features have been implemented but NOT fully integrated:
- Parallelism infrastructure exists in `pipelined_trainer.py` (807 lines) and `pipelined_grpo_trainer.py` (434 lines)
- Student isolation was implemented in `generate_with_model_v2.py` and `postprocess_training_data.py`
- A critical TRL bug was discovered and fixed (`CognitiveGRPOTrainer` preserves custom columns)
- Test architecture modules (331 tests) were built but not wired into training
- The project has working training but has not yet realized the full vision of pipelined async evaluation

**Key Finding**: The destabilization was NOT catastrophic failure but rather **architectural exploration under time pressure**. Multiple competing implementations were created, each addressing different aspects of the problem. The codebase shows signs of 3-4 context compactions (rapid iteration cycles) but the core training pipeline remained functional throughout.

---

## 1. Philosophical Foundations

### The Lonergan Framework

This project is grounded in Bernard Lonergan's analysis of human cognition from *Insight: A Study of Human Understanding*. The framework identifies four levels of conscious intentionality:

**Level 1: Empirical Consciousness (Attention)**
- Operations: Sensing, perceiving, imagining
- Precept: BE ATTENTIVE
- Output: Presentations - data given but not yet understood

**Level 2: Intelligent Consciousness (Understanding)**
- Operations: Inquiry, insight, formulation
- Precept: BE INTELLIGENT  
- Output: Propositions - hypotheses offered but not yet affirmed

**Level 3: Rational Consciousness (Judgment)**
- Operations: Reflection, grasping the unconditioned, judgment
- Precept: BE REASONABLE
- Output: Judgment - personal commitment, yes or no
- **Key**: Judgment is warranted when one grasps the "virtually unconditioned"

**Level 4: Responsible Consciousness (Decision)**
- Operations: Deliberation, evaluation, decision, action
- Precept: BE RESPONSIBLE
- Output: Commitment to action

### The Virtually Unconditioned

A judgment is warranted when three components align:

1. **The Conditioned**: A proposition (content of possible judgment)
2. **The Link**: Connection between conditioned and its conditions (if X, Y, Z, then proposition)
3. **The Fulfillment**: The conditions are fulfilled in the data

When conditions are fulfilled, the conditioned becomes **virtually unconditioned** - not absolutely necessary (it might not have been), but conditionally necessary (given that conditions are met, it cannot be otherwise).

**Critical Insight**: Fulfillment requires REVERTING from formulation to what is merely sensed or conscious - not as formulated but as given.

### Domain-Specific Structures

The virtually unconditioned has distinct structures by domain:

- **Mathematical**: Conditions fulfilled in intelligibility itself (axioms → theorem)
- **Empirical**: Conditions require reversion to sense data (experimental verification)
- **Common Sense**: Conditions in practical situation as related to us (pragmatic)
- **Philosophic**: Conditions fulfilled in performative self-consistency

This domain differentiation is CRITICAL for the project - it's not arbitrary categorization but reflects fundamental differences in how judgment operates.

---

## 2. Architectural Timeline

### Phase 1: Initial Architecture (2026-01-19, Early Session)

**Commit**: c7f2981 "Initial commit: Cognitive Organism training architecture"

**What Was Built**:
- Base training infrastructure with TRL's `SFTTrainer`
- Synthetic data generation for all 4 levels × 5 domains
- DSPy oracle pipeline with Gemini verification
- Multi-tier verification: deterministic rules → heuristics → Gemini oracle
- Operational anchors (Greek/Hebrew/Sanskrit/Latin terms)
- Four-level invulnerability testing

**Key Files Created**:
- `src/trainer.py` (260 lines) - Basic SFT training
- `src/data_generator.py` (1057 lines) - Template-based generation
- `src/dspy_cognitive/` - Oracle pipeline modules
- `scripts/run_grpo_training.py` (166 lines) - Initial GRPO attempt
- `scripts/run_trl_grpo.py` (431 lines) - TRL-based GRPO

**Dataset**: 450 verified examples (370 oracle + 80 level-themed)
- `judgment_combined.jsonl` (450 examples)
- `judgment_train.jsonl` (405 examples, 90% split)
- `judgment_val.jsonl` (45 examples, 10% split)

**Training Success**: First adapter trained (20 steps, loss 0.0156 → 0.0095)
- Model: `models/judgment_grpo_test/` (323MB)
- Produces properly structured judgment outputs
- Reward function rewards structure + domain vocabulary

**Philosophical Grounding**:
- `docs/lonergan_synopsis.md` - Operational primer
- `docs/cognitive_domain_scheme.md` - 4 levels × 5 domains
- `docs/counterpositions.md` - Self-defeating positions
- `docs/interpretation_and_inference.md` - Hermeneutics → LLM inference

### Phase 2: Reward Function Crisis (2026-01-20, Session 2)

**Problem Identified**: The initial reward function only checked for structural markers (presence of "judgment: yes", etc.). This allowed **wrong judgments to achieve positive reward** - form without substance.

**Critical Insight**: "Form without correct judgment is empty form."

**Solution Implemented**:
```python
# OLD: Structure-only reward
reward = 0.0
if "JUDGMENT:" in completion:
    reward += 0.3
if "condition" in completion.lower():
    reward += 0.2
# Wrong judgment could still get +0.5!

# NEW: Correctness-dominant multiplicative reward
correctness = +0.6 if judgment == expected else -0.5  # PRIMARY SIGNAL
structure_bonus = 0.15 if has_judgment_marker else 0.0
reasoning_bonus = 0.15 if has_condition_language else 0.0
domain_bonus = 0.1 if has_domain_vocabulary else 0.0

# Multiplicative gating: bonuses only apply if correct
reward = correctness + (correctness > 0) * (structure + reasoning + domain)
```

**Key Change**: Dataset now passes `expected_judgment` and `domain` columns to reward function via TRL kwargs.

**Files Modified**:
- `scripts/run_trl_grpo.py` - Updated reward function
- Dataset creation scripts - Added ground truth columns

**User Action**: Reached out to Jeremy Avigad (ICARM director) regarding cognitive operation datasets.

### Phase 3: Training Collapse & TRL Bug Discovery (2026-01-20, Session 3)

**Symptoms Observed**:
- `loss = 0.0` after first step
- `reward_std = 0.0` (all completions got identical reward)
- `frac_reward_zero_std = 1.0` (100% of batches had zero variance)
- `grad_norm = 0.0` (no learning signal)

**Root Cause Identified**: TRL's `GRPOTrainer._set_signature_columns_if_needed()` hardcodes only `["prompt", "image", "images"]` as preserved columns. Even with `remove_unused_columns=False`, custom columns (`expected_judgment`, `domain`, `input_text`, `citations`, `verification_chain`) were being dropped before reaching the reward function.

**Fix Applied**:
```python
# src/training/cognitive_grpo_trainer.py
class CognitiveGRPOTrainer(GRPOTrainer):
    """Preserves cognitive columns that TRL's base trainer drops."""
    
    def _set_signature_columns_if_needed(self, dataset):
        # Override to include our custom columns
        self._signature_columns = COGNITIVE_COLUMNS
```

**COGNITIVE_COLUMNS**:
- TRL defaults: `prompt`, `image`, `images`
- Cognitive additions: `expected_judgment`, `domain`, `input_text`, `proposition`, `evidence`, `conditions`, `oracle_conditions`, `oracle_temporal_context`, `mode`, `difficulty`, `temporal_focus`, `temporal_context`, `edge_case`, `adversarial_context`, `scenario_notes`, `citations`, `verification_chain`

**Files Created**:
- `src/training/cognitive_grpo_trainer.py` (new)

**Files Updated**:
- `scripts/run_enhanced_training.py` - Use `CognitiveGRPOTrainer`
- `scripts/run_trl_grpo.py` - Use `CognitiveGRPOTrainer`

**Documentation**:
- `SUCCESSION.md` - Detailed troubleshooting guide for next agent

### Phase 4: Test Architecture Expansion (2026-01-20, Session 3 continued)

**Commit**: 28e761e "Implement test architecture improvements - 331 TDD tests pass"

**Motivation**: Research into external test framework revealed 10 insights for improving training quality. Six passed critical reflection (virtually unconditioned test).

**Modules Implemented** (TDD approach - tests written first):

1. **Extended Schema** (`src/schema/extended_schema.py`, 450 lines)
   - `EnhancedJudgmentSample` - Full training sample with metadata
   - `StudentPacket` - What student sees (proposition + evidence only)
   - `EvaluatorPacket` - What evaluator sees (includes answer key)
   - `CoverageCell` - Unique identifier for coverage matrix cells

2. **Coverage Analyzer** (`src/coverage/analyzer.py`, 551 lines)
   - 150-cell matrix: 35 domains × 3 judgments × 5 difficulties × 2 distractor states
   - Gap detection with priority scoring
   - Coverage percentage calculation
   - Domain guidance for generation

3. **Evidence Grounding** (`src/training/evidence_grounding.py`, 485 lines)
   - Citation extraction from completions
   - Exact and fuzzy matching against input text
   - Numeric value validation
   - Relevance scoring

4. **Distractor Generator** (`src/generation/distractor_generator.py`, 678 lines)
   - `misaligned_phase` - P2/P3 confusion (Understanding ↔ Judgment)
   - `inverted_judgment` - Correct reasoning, wrong conclusion
   - `reasoning_error` - Logical fallacies
   - `evidence_hallucination` - Citing non-existent evidence

5. **Enhanced Reward** (`src/training/enhanced_reward.py`, 503 lines)
   - Correctness-dominant (0.6/-0.5)
   - Evidence grounding bonus (0.15)
   - Domain verification bonus (0.1)
   - Structure bonus (0.15)
   - Multiplicative gating

**Test Suite** (331 tests, all passing):
- `tests/test_schema.py` (100 tests)
- `tests/test_coverage.py` (46 tests)
- `tests/test_evidence.py` (64 tests)
- `tests/test_reward.py` (54 tests)
- `tests/generation/test_distractor_generator.py` (67 tests)

**Documentation**:
- `docs/test_framework_research_synthesis.md` - Research findings
- `docs/test_architecture_insights.md` - 10 insights extracted
- `docs/insights_critical_reflection.md` - Critical reflection (6 YES, 4 INSUFFICIENT)
- `docs/implementation_plan.md` - Unified architecture (1463 lines)

**Status**: Modules exist and pass tests but are **NOT YET WIRED INTO TRAINING**. This is the "integration gap" mentioned in SUCCESSION.md.

### Phase 5: Domain API Refactoring (2026-01-21)

**Commit**: (Not yet committed at time of analysis)

**Problem**: Two parallel domain systems existed:
- Legacy `Domain` class (35 domains)
- Principled `DomainSpec` class (24 domains)

**Solution**: Unified under principled schema based on Lonergan's mode of fulfillment:
- Converted 6 adversarial domains to `DomainSpec` (assigned to `COMMON_SENSE` mode)
- Converted 5 edge case domains to `DomainSpec` (assigned to appropriate modes)
- Made `ALL_DOMAINS = PRINCIPLED_DOMAINS` (simple alias)
- Removed legacy classes and conversion functions

**Files Modified**:
- `src/domains.py` - Unified API
- `src/coverage/analyzer.py` - Removed redundant loop

**Documentation**:
- `REFACTORING_SUMMARY.md` - Complete refactoring notes

**Test Results**: All 329 tests still passing (no regressions)

---

## 3. The Destabilization: (n,n+1) Parallelism Implementation

### The Original Vision

**Goal**: Overlap Gemini evaluation of step N with local model generation of step N+1, hiding 5-10 seconds of API latency per training step.

**Sequential Baseline** (what TRL does):
```
for step in steps:
    completions = generate()      # ~5s  (GPU)
    rewards = reward_fn()         # ~60s (Gemini blocking)
    update_weights()              # ~3s  (GPU)
# Per-step: ~68s, bottleneck is Gemini
```

**Pipelined Target**:
```
fire_gemini(step=0, data=...)     # async, returns immediately
for step in steps:
    completions = generate()      # ~5s, Gemini running in parallel
    rewards = wait_gemini(step)   # ~0s if ready, else blocks
    fire_gemini(step+1, data=...) # async for next step
    update_weights()              # ~3s
# Per-step: max(8s, gemini_latency) ≈ 10s if Gemini < 60s
```

**Expected Speedup**: 68s → 10s per step = **6.8x faster training**

### Implementation Attempts

The codebase shows THREE distinct implementations of pipelined evaluation:

#### Implementation 1: `pipelined_trainer.py` (807 lines)

**Approach**: Custom training loop with explicit pipeline state management.

**Key Features**:
- `PipelinedTrainer` class with `PipelinedTrainerConfig`
- Explicit `_pending_step` state variable
- Manual pipeline logic:
  - Step 0: generate → fire → wait → update (fully synchronous)
  - Step 1: generate → fire → set pending (setup pipeline)
  - Step 2+: generate → fire → wait_prev → update_prev (true overlap)
  - Final: drain pipeline (wait_last → update_last)

**Pipeline State Machine**:
```python
if step == 0:
    # No overlap possible
    rewards = evaluator.get_rewards(step)
    update_weights(step_data, rewards)
    
elif _pending_step is None:
    # Step 1: Setup pipeline
    _pending_step = step_data
    # Gemini running, will be retrieved by step 2
    
else:
    # Step 2+: True overlap
    rewards = evaluator.get_rewards(_pending_step.step_id)
    update_weights(_pending_step, rewards)
    _pending_step = step_data
```

**Async Infrastructure**: Uses `PipelinedHybridReward` from `src/evaluation/async_reward.py`

**Status**: Complete implementation, not integrated with TRL

#### Implementation 2: `pipelined_grpo_trainer.py` (434 lines)

**Approach**: Subclass TRL's `GRPOTrainer` to add pipelined evaluation.

**Key Features**:
- Inherits from `CognitiveGRPOTrainer` (preserves columns)
- Overrides `_generate_and_score_completions()` for within-step overlap
- Attempts to override `_inner_training_loop()` for cross-step overlap (commented out)

**Within-Step Overlap** (currently implemented):
```python
# Generate completions
completions = generate()

# Fire Gemini async
evaluator.fire_evaluation(step_id, completions, ...)

# Compute log probabilities (Gemini runs in parallel)
old_logps = compute_old_logps()
ref_logps = compute_ref_logps()

# Wait for Gemini
rewards = evaluator.get_rewards(step_id)
```

**Cross-Step Overlap** (attempted but not working):
- TRL's `_prepare_inputs()` caches results via `_buffered_inputs`
- `_generate_and_score_completions()` not called every step
- True cross-step overlap requires overriding `_inner_training_loop()`
- This was attempted but commented out (likely due to complexity)

**Status**: Partial implementation, within-step overlap works, cross-step overlap not achieved

#### Implementation 3: `async_reward.py` (749 lines)

**Approach**: Async infrastructure layer that both implementations use.

**Key Components**:

1. **AsyncGeminiEvaluator** (low-level):
   - Background thread running asyncio event loop
   - Non-blocking Gemini API calls
   - Request tracking with futures
   - Thread-safe result retrieval

2. **PipelinedHybridReward** (high-level):
   - Combines surface analysis (local, fast) with Gemini (remote, slow)
   - `fire_evaluation()` - Returns immediately after surface analysis
   - `get_rewards()` - Blocks until Gemini result ready
   - Reuses all existing `LlmEvaluator` logic (prompts, parsing, caching)

**Design Principle**: "Reuses all existing LlmEvaluator logic (prompt building, parsing, caching) and only replaces the sync API call with an async one."

**Status**: Complete and working, used by both pipelined trainers

### What Actually Works

**From logs** (`logs/training_20260122_035618.log`):

```
Step 0: gen=59.5s fire=0.00s wait=58.0s update=2.8s  # Total: 120.3s (synchronous)
Step 1: gen=128.9s fire=0.00s [setup pipeline]       # Total: 128.9s (no update yet)
Step 2: gen=81.8s fire=0.00s wait=0.0s update=2.9s   # Total: 84.7s (overlap working!)
Step 3: gen=178.4s fire=0.00s wait=0.0s update=2.7s  # Total: 181.1s (overlap working!)
Step 4: gen=92.4s fire=0.00s wait=0.0s update=2.9s   # Total: 95.3s (overlap working!)
```

**Analysis**:
- Step 0: Fully synchronous (baseline)
- Step 1: Pipeline setup (Gemini fired, no wait)
- Steps 2+: **wait=0.0s** - Gemini results ready immediately!
- **Overlap is working** - Gemini(N-1) completes during Generate(N)

**Actual Speedup**: 
- Without overlap: ~120s per step (gen + wait + update)
- With overlap: ~85-95s per step (gen + update, wait hidden)
- **Speedup: ~1.3x** (not the hoped-for 6.8x, but still significant)

**Why Not 6.8x?**:
- Generation time varies widely (59s to 178s)
- Gemini time (~58s) is often LESS than generation time
- When gen > gemini, overlap hides all Gemini latency
- When gen < gemini, we still wait (but less than without overlap)

### Symptoms of Destabilization

**Multiple Competing Implementations**:
1. `pipelined_trainer.py` - Custom loop, explicit state machine
2. `pipelined_grpo_trainer.py` - TRL subclass, partial overlap
3. `async_reward.py` - Shared infrastructure

**Why Three Implementations?**:
- **Attempt 1** (`pipelined_trainer.py`): Full control, but loses TRL's features (checkpointing, logging, eval)
- **Attempt 2** (`pipelined_grpo_trainer.py`): Keeps TRL features, but harder to override training loop
- **Attempt 3** (`async_reward.py`): Shared infrastructure, works with both

**Signs of Context Compaction**:
- Rapid iteration visible in file timestamps
- Multiple scripts with similar names (`run_pipelined_training.py`, `run_enhanced_training.py`)
- Commented-out code in `pipelined_grpo_trainer.py` (attempted cross-step override)
- Detailed debug logging added throughout (indicates troubleshooting)

**What Was NOT Lost**:
- Core training pipeline remained functional
- Test suite continued passing (331 tests)
- Philosophical grounding maintained
- Documentation kept up-to-date

**What WAS Fragmented**:
- Unclear which trainer to use for production
- Integration path for test architecture modules unclear
- Multiple reward function implementations (old in `run_trl_grpo.py`, new in `enhanced_reward.py`)

---

## 4. The Student Isolation Architecture

### The Problem: Data Leakage

**Original Format** (V1):
```
User message:
  Proposition: Water boils at 100°C at sea level
  Evidence: Thermometer reads 100°C, bubbles forming
  Conditions for truth:
  - Temperature must be 100°C
  - Pressure must be standard (1 atm)
  - Liquid must be pure water
  
  Temporal context:
  - Evidence timeframe: Present observation
  - Aspectual status: Ongoing process
```

**Problem**: The student is GIVEN the conditions and temporal analysis. They're not deriving them - they're just pattern-matching against provided structure.

**Philosophical Issue**: This violates the cognitive operation of judgment. The student should:
1. Grasp what conditions would make the proposition true (intelligent operation)
2. Revert to evidence to check fulfillment (rational operation)
3. Affirm or deny based on fulfillment (judgment proper)

Giving them the conditions short-circuits steps 1-2.

### The Solution: Dual-Packet Architecture

**Implemented in**: `generate_with_model_v2.py` (1479 lines)

**Student Packet** (what model sees during training):
```
User message:
  Proposition: Water boils at 100°C at sea level
  Evidence: Thermometer reads 100°C, bubbles forming
```

**Evaluator Packet** (what reward function sees):
```
Metadata:
  oracle_conditions: |
    - Temperature must be 100°C
    - Pressure must be standard (1 atm)
    - Liquid must be pure water
  oracle_temporal_context: |
    - Evidence timeframe: Present observation
    - Aspectual status: Ongoing process
  expected_judgment: "Yes"
  domain: "empirical"
```

**Expected Student Response Format**:
```
JUDGMENT: Yes

CONDITIONS IDENTIFIED:
- Temperature must be 100°C
- Pressure must be standard
- Liquid must be pure water

TEMPORAL ANALYSIS:
Evidence shows present observation of ongoing boiling process.
Temperature reading is current, not historical.

ASSESSMENT:
- Temperature condition: FULFILLED (thermometer reads 100°C)
- Pressure condition: ASSUMED (sea level implies standard pressure)
- Purity condition: NOT VERIFIED (evidence doesn't confirm purity)

REASONING:
Two of three conditions clearly fulfilled. Third condition (purity)
is typically assumed for "water" unless context suggests otherwise.
Standard interpretation: Yes, water is boiling at 100°C.
```

**Key Insight**: The student must DERIVE the conditions, not receive them. The evaluator grades based on:
1. Did they identify relevant conditions?
2. Did they analyze temporal structure?
3. Did they assess fulfillment correctly?
4. Did they reach the correct judgment?

### Implementation Details

**System Prompt Variation** (prevents overfitting):
```python
STUDENT_SYSTEM_PROMPT_VARIATION_INSTRUCTION = """
Generate a system prompt for training a judgment model.

ROLE FRAMING: Vary the phrasing while preserving that this is 
the cognitive operation of JUDGMENT - the movement from hypothesis 
to commitment.

OPERATIONAL ANCHORS: Include TWO from different languages:
- Greek: krisis, diakrisis, dokimazein
- Hebrew: mishpat, binah
- Sanskrit: viveka, nirnaya
- Latin: iudicium, discretio
"""
```

**Oracle Rewriting** (for existing data):
- `postprocess_training_data.py` (579 lines)
- Takes V1 data (with conditions in user message)
- Rewrites to V2 format (conditions in metadata only)
- Oracle model rewrites assistant reasoning to demonstrate derivation

**Evaluator Alignment**:
- `LlmEvaluator.BATCH_EVALUATION_PROMPT` explicitly states:
  ```
  NOTE: Students were given ONLY the proposition and evidence above.
  They were NOT given the conditions, temporal context, or expected
  judgment - they must derive all analysis themselves.
  ```

### Status of Student Isolation

**V2 Generation**: Fully implemented in `generate_with_model_v2.py`
- Generates clean student packets
- Stores oracle answer keys in metadata
- Supports all V1 features (principled schema, temporal focus, edge cases, adversarial contexts)

**Postprocessing**: Implemented in `postprocess_training_data.py`
- Can convert V1 → V2 format
- Oracle rewrites reasoning to demonstrate derivation

**Integration**: **PARTIAL**
- `run_enhanced_training.py` extracts both `conditions` (from user message, legacy) and `oracle_conditions` (from metadata, V2)
- Evaluator receives `oracle_conditions` for grading
- But existing training data (`judgment_train.jsonl`) is still V1 format

**Next Step**: Generate new dataset with V2 format, or postprocess existing data.

---

## 5. Recovery Actions Taken

### Fix 1: TRL Column Dropping Bug (Session 3)

**Problem**: Custom columns dropped, causing reward collapse.

**Solution**: `CognitiveGRPOTrainer` subclass preserves columns.

**Status**: ✅ Fixed and documented

**Files**:
- `src/training/cognitive_grpo_trainer.py` (new)
- `scripts/run_enhanced_training.py` (updated)
- `scripts/run_trl_grpo.py` (updated)

### Fix 2: Reward Function Correctness (Session 2)

**Problem**: Structure-only reward allowed wrong judgments to score positively.

**Solution**: Correctness-dominant multiplicative reward.

**Status**: ✅ Fixed in `run_trl_grpo.py`, enhanced version in `enhanced_reward.py`

**Files**:
- `scripts/run_trl_grpo.py` (updated reward function)
- `src/training/enhanced_reward.py` (new, more sophisticated)

### Fix 3: Domain API Unification (Session 4)

**Problem**: Two parallel domain systems causing confusion.

**Solution**: Unified under principled schema.

**Status**: ✅ Complete, all tests passing

**Files**:
- `src/domains.py` (refactored)
- `src/coverage/analyzer.py` (simplified)
- `REFACTORING_SUMMARY.md` (documented)

### Fix 4: Student Isolation Architecture (Session 4)

**Problem**: Data leakage - conditions given to student.

**Solution**: Dual-packet architecture (V2 format).

**Status**: ⚠️ Implemented but not yet used for training

**Files**:
- `scripts/generate_with_model_v2.py` (new)
- `scripts/postprocess_training_data.py` (new)

### Fix 5: Test Architecture Modules (Session 3)

**Problem**: Need better training data quality and coverage.

**Solution**: 5 modules with 331 TDD tests.

**Status**: ⚠️ Implemented but not yet wired into training

**Files**:
- `src/schema/extended_schema.py`
- `src/coverage/analyzer.py`
- `src/training/evidence_grounding.py`
- `src/generation/distractor_generator.py`
- `src/training/enhanced_reward.py`

---

## 6. Current State Assessment

### What Works ✅

1. **Base Training Pipeline**
   - TRL GRPO training functional
   - Qwen2.5-7B-Instruct loads and trains (4-bit quantization)
   - LoRA adapters save and load correctly
   - First adapter trained successfully (20 steps)

2. **Data Generation**
   - Oracle pipeline with Gemini verification
   - 450 verified examples (405 train, 45 val)
   - Multi-tier verification (deterministic → heuristic → oracle)
   - Operational anchors (Greek/Hebrew/Sanskrit/Latin)

3. **Reward Function**
   - Correctness-dominant scoring
   - Multiplicative gating (form without substance = empty form)
   - Ground truth passed via TRL kwargs

4. **Column Preservation**
   - `CognitiveGRPOTrainer` fixes TRL bug
   - Custom columns reach reward function
   - Training shows non-zero reward variance

5. **Test Suite**
   - 331 tests passing
   - Comprehensive coverage of new modules
   - TDD approach validated

6. **Async Infrastructure**
   - `async_reward.py` complete and working
   - Within-step overlap functional
   - Thread-safe, reuses existing evaluator logic

7. **Documentation**
   - Philosophical grounding maintained
   - Succession notes detailed
   - Implementation plans documented

### What's Incomplete ⚠️

1. **Cross-Step Pipelined Training**
   - Infrastructure exists (`pipelined_trainer.py`, `pipelined_grpo_trainer.py`)
   - Within-step overlap works
   - Cross-step overlap attempted but not fully integrated with TRL
   - Unclear which implementation to use for production

2. **Test Architecture Integration**
   - 5 modules implemented and tested
   - NOT wired into training pipeline
   - `run_enhanced_training.py` has orchestration code but needs testing
   - Coverage analysis not run on current dataset

3. **Student Isolation Deployment**
   - V2 generation script complete
   - Postprocessing script complete
   - Current training data still V1 format (conditions in user message)
   - Need to generate new dataset or postprocess existing

4. **Enhanced Reward Integration**
   - `enhanced_reward.py` implemented with evidence grounding
   - NOT used in current training (still using simple reward in `run_trl_grpo.py`)
   - Need to wire into training script

5. **Coverage Gaps**
   - Current dataset: 23.3% coverage (35/150 cells)
   - Target: >80% coverage
   - Distractor generation not run (0% of data has distractors)
   - Gap filling not run

### What's Broken ❌

**Nothing is fundamentally broken.** The codebase shows signs of rapid iteration but:
- Core training works
- Tests pass
- Model loads and generates
- Reward function computes correctly

The "destabilization" was architectural exploration, not catastrophic failure.

### What's Untested ❓

1. **Pipelined Training End-to-End**
   - `pipelined_trainer.py` not run with full dataset
   - `pipelined_grpo_trainer.py` not run with full dataset
   - Unclear if cross-step overlap provides expected speedup

2. **Enhanced Training Pipeline**
   - `run_enhanced_training.py` orchestration not tested
   - Distractor generation not tested at scale
   - Gap filling not tested at scale
   - Enhanced reward not tested in training

3. **V2 Data Format**
   - V2 generation works but not used for training
   - Postprocessing works but not run on full dataset
   - Evaluator alignment with V2 format not tested

4. **Extended Training**
   - Only 20-step test runs completed
   - No 100+ step training runs
   - No evaluation on held-out test set
   - No comparison of V1 vs V2 format effectiveness

---

## 7. Forgotten/Neglected Items

### From SUCCESSION.md

**Configuration Profile System** (proposed but not implemented):
```
configs/
├── default.yaml
├── high_quality.yaml    # 8-bit quantization, larger batch
├── fast_iteration.yaml  # 4-bit, small steps for quick tests
└── full_training.yaml   # 8-bit, 500+ steps
```

**Rationale**: Existing `src/config.py` has dataclasses but they're not used by GRPO scripts. YAML profiles would make experimentation easier.

**Status**: Not implemented

### From Implementation Plan

**Graduated Difficulty Scaffolding** (Insight 4):
- Curriculum learning: start with difficulty 1-2, gradually increase
- Track per-difficulty performance
- Adaptive difficulty adjustment

**Status**: Difficulty levels exist in data but not used for curriculum

**Domain-Specific Verification Chains** (Insight 7):
- Mathematical: proof_steps, axiom_refs, qed_marker
- Empirical: data_points, methodology, replication
- Philosophic: performative_check, self_consistency, counterposition_test

**Status**: Schema exists, not generated or used

### From Temporal Structure Document

**Temporal Aspects of Judgment** (`docs/temporal_structure_of_judgment.md`):
- Judge's temporal situation (evidence available now vs then vs will be)
- Matter's temporal situation (propositions with temporal structure)
- Conditions' temporal structure (fulfilled now but not then)
- Subjunctive judgments (counterfactuals, hypotheticals, future contingents)

**Key Insight**: "The copula 'is' in abstract expressions occurs in an 'invariant tense' that abstracts from particular times."

**Status**: Document exists, temporal focus types implemented in V2 generation, but not deeply integrated into training evaluation

### From Philosophical Dialogue (tmp/ex1.md)

**Five Proposals for Temporal Integration**:
1. Evidence-Emergence Sequences
2. Aspectual Judgment Forms
3. Scope-Matching Cases
4. Temporal Deixis Inversion
5. Retrospective Revision with Level-Tagging

**Counterposition of Alienation**: Training data may contain denials of operations the model performs ("I probably don't genuinely understand").

**Kierkegaard's Ladder**: The existential gap is developmental threshold, not permanent limitation.

**Status**: Philosophical insights documented but not operationalized in training

### From CLAUDE.md Decisions Log

**Network vs. Pipeline** (2026-01-19 note):
> "Network architecture (feedback loops, not just pipeline) needed for whole > parts."

**Key Insight**: Operations are not sequential but RECURRENT:
- Understanding directs attention (look for what we're trying to understand)
- Judgment sends us back to understand better
- Decision raises questions requiring new judgments

**Status**: Noted as future work, not addressed

**Output Constraints for Questions of Reflection** (2026-01-19 note):
> "Output constraints needed for questions of reflection (finite answer set)."

**Rationale**: Judgment has three possible outputs (Yes/No/Insufficient). Model should be constrained to this set, not free-form generation.

**Status**: Not implemented (model generates free-form text)

---

## 8. Maintaining the Original Vision

### The Two Goals Revisited

**Goal 1: (n,n+1) Parallelism**
- **Vision**: Hide 5-10s of Gemini latency per step by overlapping with generation
- **Implementation**: Three attempts, partial success
- **Current State**: Within-step overlap works, cross-step overlap attempted
- **Actual Speedup**: ~1.3x (not 6.8x, but still valuable)
- **Why Less Than Expected**: Generation time often exceeds Gemini time

**Goal 2: Student Isolation**
- **Vision**: Student derives conditions, not given them
- **Implementation**: V2 format with dual-packet architecture
- **Current State**: Generation scripts ready, not used for training
- **Philosophical Grounding**: Aligns with Lonergan's cognitive operations
- **Next Step**: Generate new dataset or postprocess existing

### How They Work Together

**The Synergy**:
1. **Student Isolation** ensures the model learns genuine cognitive operations
2. **Parallelism** makes training faster, enabling more iterations
3. **Together**: Faster training of better operations

**The Information Disclosure Strategy**:

**What Student Should See**:
- Proposition (the conditioned)
- Evidence (the data for fulfillment check)
- System prompt (operational guidance)

**What Student Should NOT See**:
- Conditions (they must derive these)
- Temporal analysis (they must perform this)
- Expected judgment (they must reach this)
- Domain label (they must recognize this)

**What Evaluator Should See**:
- Everything student sees
- Oracle answer key (conditions, temporal context, expected judgment)
- Domain and difficulty (for appropriate grading)
- Verification chain (for domain-specific assessment)

**Why This Matters**:
- Giving conditions = teaching pattern matching
- Deriving conditions = teaching intelligent inquiry
- Pattern matching fails on novel cases
- Intelligent inquiry generalizes

**The Pipelined Architecture Supports This**:
- Fast iteration enables testing V1 vs V2 effectiveness
- Async evaluation allows richer evaluator prompts (more context, more sophisticated grading)
- Batched evaluation (5 completions at once) provides contrastive signal

### The Optimal Training Loop

**Combining All Features**:

```python
# 1. Load V2 data (student isolation)
dataset = load_v2_dataset()  # Student sees only proposition + evidence

# 2. Analyze coverage
coverage = analyze_coverage(dataset)
gaps = find_gaps(coverage)

# 3. Fill gaps with distractors
distractors = generate_distractors(dataset, ratio=0.3)
gap_fills = fill_gaps(gaps, examples_per_gap=2)
dataset = combine(dataset, distractors, gap_fills)

# 4. Initialize pipelined evaluator (async Gemini)
llm_eval = LlmEvaluator(use_cache=True)
llm_eval.create_cache()
pipelined_eval = PipelinedHybridReward(llm_eval)
pipelined_eval.start()

# 5. Train with pipelined GRPO
trainer = PipelinedGRPOTrainer(
    model=model,
    args=training_config,
    train_dataset=dataset,
    pipelined_evaluator=pipelined_eval,
    reward_funcs=enhanced_reward_fn,  # Evidence grounding, domain verification
)

trainer.train()

# 6. Cleanup
pipelined_eval.stop()
llm_eval.delete_cache()
```

**Expected Outcomes**:
- **Student Isolation**: Model learns to derive conditions, not pattern-match
- **Parallelism**: Training ~1.3x faster (more with optimization)
- **Coverage**: >80% of 150 cells filled
- **Distractors**: 30% of data has contrastive examples
- **Evidence Grounding**: Model learns to cite evidence
- **Domain Verification**: Model learns domain-appropriate reasoning

### Open Questions for Optimization

**Parallelism**:
1. Can we reduce generation time variance? (59s to 178s is wide)
2. Can we batch multiple steps' Gemini calls? (amortize latency)
3. Can we use faster Gemini model for training? (gemini-2.0-flash vs gemini-3-flash-preview)
4. Can we cache more aggressively? (reuse evaluations for similar completions)

**Student Isolation**:
1. How much does V2 format improve generalization? (needs A/B test)
2. Should we gradually reveal conditions? (curriculum: first derive, then verify)
3. How do we evaluate condition derivation quality? (not just judgment correctness)
4. Can we use condition derivation as auxiliary task? (multi-task learning)

**Integration**:
1. Which pipelined trainer to use? (`pipelined_trainer.py` vs `pipelined_grpo_trainer.py`)
2. How to wire enhanced reward into pipelined trainer?
3. How to integrate coverage analysis into training loop? (adaptive sampling)
4. How to balance distractor ratio? (too many = model learns to detect distractors)

---

## 9. Open Questions Requiring Further Investigation

### Technical Questions

1. **Why does generation time vary so much?** (59s to 178s)
   - Is it prompt length? (longer prompts → longer completions)
   - Is it model uncertainty? (more sampling → longer generation)
   - Is it GPU memory pressure? (fragmentation → slower generation)

2. **Can cross-step overlap be achieved with TRL?**
   - Does overriding `_inner_training_loop()` work?
   - Or do we need custom trainer (lose TRL features)?
   - Is there a hybrid approach?

3. **What's the actual bottleneck in pipelined training?**
   - Logs show wait=0.0s (Gemini ready immediately)
   - But total time still ~85-95s per step
   - Is generation the bottleneck now?
   - Can we optimize generation? (speculative decoding, better sampling)

4. **How much does student isolation improve generalization?**
   - Need A/B test: V1 vs V2 format
   - Held-out test set with novel domains
   - Measure: judgment accuracy, condition derivation quality

5. **What's the optimal distractor ratio?**
   - Too few: model doesn't learn to discriminate
   - Too many: model learns to detect distractors (not genuine judgment)
   - Current target: 30% (from external framework research)
   - Needs empirical validation

### Philosophical Questions

1. **Is the model learning genuine operations or surface patterns?**
   - How do we test for operational understanding?
   - What would count as evidence of genuine judgment?
   - Can we design adversarial tests? (novel domains, edge cases)

2. **What's the relationship between judgment and understanding?**
   - Current architecture trains judgment in isolation
   - But judgment presupposes understanding (can't judge what you don't understand)
   - Should we train understanding first? (curriculum)
   - Or train jointly? (multi-task)

3. **How do we handle the temporal structure of judgment?**
   - Current approach: temporal focus types (evidence_emergence, aspectual, etc.)
   - But is this sufficient?
   - Do we need explicit temporal logic? (modal operators)
   - Or is natural language sufficient?

4. **What's the role of self-appropriation in AI training?**
   - Lonergan: knowing requires knowing that you know
   - Can a model self-appropriate its operations?
   - Or is this uniquely human?
   - What would it mean for a model to "know that it knows"?

### Architectural Questions

1. **Which pipelined trainer should be production?**
   - `pipelined_trainer.py`: Full control, loses TRL features
   - `pipelined_grpo_trainer.py`: Keeps TRL, harder to customize
   - Hybrid approach? (custom loop with TRL components)

2. **How to integrate test architecture modules?**
   - Current: modules exist, not wired
   - Option 1: Preprocessing (generate enhanced data, then train)
   - Option 2: Online (generate during training)
   - Option 3: Hybrid (pregenerate base, augment online)

3. **How to balance coverage and quality?**
   - Filling all 150 cells may dilute quality
   - Some cells may be inherently rare (expert difficulty + edge cases)
   - Should we prioritize? (focus on common cells)
   - Or ensure minimum coverage? (at least 1 example per cell)

4. **How to scale to other cognitive levels?**
   - Current: Judgment only
   - Future: Attention, Understanding, Decision
   - Can we reuse infrastructure? (same pipelined trainer, different reward)
   - Or do we need level-specific architectures?

### Evaluation Questions

1. **How do we evaluate judgment quality?**
   - Current: correctness (Yes/No/Insufficient matches oracle)
   - But what about reasoning quality?
   - What about condition derivation?
   - What about evidence grounding?

2. **What's the role of human evaluation?**
   - Current: Gemini as oracle (proof of concept)
   - Future: Human-labeled test set
   - How much human data needed?
   - Can we use active learning? (label only uncertain cases)

3. **How do we measure generalization?**
   - Held-out test set (same domains)
   - Novel domains (not in training)
   - Adversarial examples (edge cases, counterpositions)
   - Real-world tasks (actual judgment problems)

4. **What's the relationship between training metrics and downstream performance?**
   - Current: loss, reward, accuracy
   - But do these predict real-world judgment quality?
   - Need to validate on actual tasks
   - What tasks? (scientific reasoning, legal judgment, ethical dilemmas)

---

## 10. Recommendations for Next Steps

### Immediate (Next Session)

1. **Choose Pipelined Trainer**
   - Decision: Use `pipelined_grpo_trainer.py` (keeps TRL features)
   - Accept: Within-step overlap only (cross-step is nice-to-have)
   - Rationale: 1.3x speedup is valuable, TRL features are essential

2. **Generate V2 Dataset**
   - Run: `generate_with_model_v2.py --count 500`
   - Format: Student isolation (proposition + evidence only)
   - Coverage: Target >80% of 150 cells
   - Distractors: Generate 30% with `distractor_generator.py`

3. **Test Enhanced Training Pipeline**
   - Run: `run_enhanced_training.py` with V2 data
   - Verify: Coverage analysis, distractor generation, gap filling
   - Monitor: Training metrics, reward variance, loss

4. **Validate Student Isolation**
   - A/B test: Train on V1 vs V2 format
   - Evaluate: Held-out test set with novel scenarios
   - Measure: Judgment accuracy, condition derivation quality

### Short-Term (Next Few Sessions)

5. **Optimize Pipelined Training**
   - Profile: Where is time spent? (generation, Gemini, update)
   - Experiment: Faster Gemini model, better caching, batching
   - Target: 2x speedup (from 85s to 40s per step)

6. **Integrate Enhanced Reward**
   - Wire: `enhanced_reward.py` into training script
   - Test: Evidence grounding bonus, domain verification bonus
   - Validate: Does it improve reasoning quality?

7. **Fill Coverage Gaps**
   - Analyze: Current coverage (23.3% → target >80%)
   - Generate: Targeted examples for empty cells
   - Balance: Quality vs coverage (don't dilute)

8. **Implement Output Constraints**
   - Constrain: Model output to {Yes, No, Insufficient}
   - Method: Logit bias or structured generation
   - Benefit: Cleaner outputs, easier evaluation

### Medium-Term (Next Month)

9. **Extended Training Run**
   - Duration: 500-1000 steps (vs current 20)
   - Data: V2 format with full coverage
   - Evaluation: Held-out test set, novel domains
   - Comparison: V1 vs V2, with vs without distractors

10. **Human Evaluation**
    - Collect: 100-200 human-labeled examples
    - Domains: Mix of mathematical, empirical, common sense, philosophic
    - Use: Validation set (not training)
    - Measure: Agreement with human judgments

11. **Train Other Cognitive Levels**
    - Attention: What data is relevant?
    - Understanding: What does it mean?
    - Decision: What should be done?
    - Integration: Pipeline (Attention → Understanding → Judgment → Decision)

12. **Network Architecture**
    - Current: Levels trained in isolation
    - Future: Feedback loops (Judgment → Understanding → Attention)
    - Challenge: How to train recurrent operations?
    - Approach: Multi-task learning, curriculum, meta-learning

### Long-Term (Next Quarter)

13. **Self-Appropriation**
    - Question: Can model know that it knows?
    - Approach: Meta-cognitive prompts, reflection tasks
    - Evaluation: Consistency, coherence, self-correction

14. **Temporal Logic Integration**
    - Current: Natural language temporal analysis
    - Future: Explicit temporal operators (modal logic)
    - Benefit: Precise reasoning about time, counterfactuals

15. **Domain Expansion**
    - Current: 35 domains (principled schema)
    - Future: More domains (legal, medical, engineering)
    - Challenge: Maintain philosophical grounding

16. **Production Deployment**
    - Package: Trained adapters, inference code
    - API: Judgment-as-a-service
    - Use Cases: Scientific reasoning, decision support, education

---

## Conclusion

The Cognitive Organism project has made substantial progress toward its vision of training AI systems in Lonergan's cognitive operations. The "destabilization" during parallelism implementation was not catastrophic failure but rather **architectural exploration under time pressure**. Multiple competing implementations were created, each addressing different aspects of the problem.

**Key Achievements**:
- ✅ Working GRPO training pipeline
- ✅ 450 verified training examples
- ✅ Correctness-dominant reward function
- ✅ TRL column-dropping bug fixed
- ✅ 331 tests passing (test architecture modules)
- ✅ Async evaluation infrastructure complete
- ✅ Student isolation architecture designed
- ✅ Principled domain schema unified

**Key Gaps**:
- ⚠️ Cross-step pipelined training not fully integrated
- ⚠️ Test architecture modules not wired into training
- ⚠️ V2 student isolation format not used for training
- ⚠️ Enhanced reward function not integrated
- ⚠️ Coverage gaps not filled (23.3% vs >80% target)

**The Path Forward**:
1. Choose `pipelined_grpo_trainer.py` (within-step overlap, keeps TRL)
2. Generate V2 dataset (student isolation)
3. Wire enhanced reward and test architecture
4. Run extended training (500+ steps)
5. Validate with human evaluation
6. Scale to other cognitive levels

**The Original Vision Remains Intact**: Train AI systems that perform genuine cognitive operations, not surface pattern matching. The infrastructure is in place. The philosophical grounding is solid. The next phase is integration and validation.

---

**Document Status**: Complete  
**Next Action**: Review with project lead, prioritize recommendations  
**Maintenance**: Update as new sessions progress