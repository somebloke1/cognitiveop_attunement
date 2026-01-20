# Succession Notes for Next Agent

*Last updated: 2026-01-20 (Session 2)*

## Immediate Context

**IMPORTANT**: Test architecture improvements have been implemented (331 TDD tests passing). The modules are tested in isolation but **NOT YET WIRED** into the training pipeline.

**NEXT TASK**: Build the full enhanced training pipeline (see "Successor Agent Prompt" at bottom).

## Project State Summary

The Cognitive Organism project has:

1. **Oracle Pipeline**: Gemini-based data generation with multi-tier verification ✓
2. **Dataset**: 450 verified examples (405 train, 45 val) ✓
3. **GRPO Training**: TRL-based with correctness-dominant reward ✓
4. **Test Architecture Modules** (NEW): 5 modules, 331 tests passing ✓
5. **Integration**: **NOT DONE** - modules exist but not wired into training

## Critical Change: Reward Function (2026-01-20)

The old reward function only checked for structural markers (presence of "judgment: yes", etc.). This allowed wrong judgments to achieve positive reward.

**New reward function** (`scripts/run_trl_grpo.py`):
- **Correctness**: +0.6 if judgment matches oracle, -0.5 if wrong (PRIMARY SIGNAL)
- **Structure**: +0.15 if has explicit judgment marker (GATED by correctness)
- **Reasoning**: +0.15 for condition/fulfillment language (GATED by correctness)
- **Domain**: +0.1 for domain-appropriate vocabulary (GATED by correctness)

**Multiplicative scoring**: Structural bonuses only apply if judgment is correct. Wrong judgment = 0 multiplier on bonuses. Form without correct judgment is empty form.

**Dataset now passes** `expected_judgment` and `domain` columns to reward function via TRL kwargs.

## What Just Happened

The previous agent:
- Identified critical flaw: old reward rewarded form without substance
- Implemented correctness-dominant multiplicative reward function
- Updated dataset creation to pass ground truth to reward function
- Discussed data limitations (Gemini as oracle = proof of concept, not final)
- User reached out to Jeremy Avigad (ICARM director) re: cognitive operation datasets

## Training Command (User Will Run Separately)

```bash
cd /home/dgk/projects/cognitiveop_attunement
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python scripts/run_trl_grpo.py \
  --data data/oracle_generated/judgment_train.jsonl \
  --output models/judgment_grpo_extended \
  --steps 100 \
  --batch-size 1
```

## Potential Issues and Solutions

### 1. OOM (Out of Memory)
**Symptom**: CUDA out of memory error
**Solution**:
- Reduce batch size to 1 (already set)
- Reduce `num_generations` in GRPOConfig (currently 4)
- Reduce `max_completion_length` (currently 512)

### 2. Training Stalls
**Symptom**: Progress stops, no error
**Solution**: Check GPU utilization with `nvidia-smi`. May need to reduce gradient accumulation.

### 3. Reward Stays Low or Negative
**Symptom**: Reward doesn't improve or stays negative
**Context**: With the new correctness-dominant reward function, wrong judgments get -0.5. Early in training, rewards may be negative if the model frequently judges incorrectly.
**Expected behavior**: As model learns, correct judgments should increase, bringing average reward toward +0.6 to +1.0.
**If stuck at negative**: The model isn't learning to match oracle judgments. Check:
- Are `expected_judgment` columns being passed? (Debug print in reward function)
- Is the judgment extraction regex working? (Test with sample completions)
- Is the model outputting judgment markers at all?

### 4. Loss Spikes
**Symptom**: Loss suddenly increases
**Solution**: Learning rate may be too high. Edit `GRPOConfig` in script to lower from 1e-5.

## Key Files to Know

| File | Purpose |
|------|---------|
| `scripts/run_trl_grpo.py` | Main training script |
| `data/oracle_generated/judgment_train.jsonl` | Training data (450 examples) |
| `models/judgment_grpo_test/` | First successful adapter |
| `CLAUDE.md` | Project state and philosophy |

### New Test Architecture Modules (2026-01-20)

| Module | Purpose |
|--------|---------|
| `src/schema/extended_schema.py` | EnhancedJudgmentSample, StudentPacket, EvaluatorPacket, CoverageCell |
| `src/coverage/analyzer.py` | 150-cell coverage matrix, gap detection |
| `src/training/evidence_grounding.py` | Citation extraction and validation |
| `src/generation/distractor_generator.py` | P2/P3 distractors for contrastive learning |
| `src/training/enhanced_reward.py` | Multiplicative gating reward (use THIS, not run_trl_grpo.py's) |

### Documentation

| File | Purpose |
|------|---------|
| `docs/test_framework_research_synthesis.md` | Research findings from external framework |
| `docs/test_architecture_insights.md` | 10 insights extracted |
| `docs/insights_critical_reflection.md` | Critical reflection (6 YES, 4 INSUFFICIENT) |
| `docs/implementation_plan.md` | Unified implementation plan |

## How to Test the Trained Adapter

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python3 << 'EOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

model_name = "Qwen/Qwen2.5-7B-Instruct"
adapter_path = "models/judgment_grpo_extended"  # or judgment_grpo_test

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config, device_map="auto"
)
model = PeftModel.from_pretrained(model, adapter_path)

# Test inference
messages = [
    {"role": "system", "content": "You are a Judgment cognitive operation..."},
    {"role": "user", "content": "Proposition: [test proposition]..."}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(output[0], skip_special_tokens=True))
EOF
```

## Pending Improvements

### Configuration Profile System
Create a YAML-based configuration profile system for training:

**Location**: `configs/`

**Proposed structure**:
```
configs/
├── default.yaml
├── high_quality.yaml    # 8-bit quantization, larger batch
├── fast_iteration.yaml  # 4-bit, small steps for quick tests
└── full_training.yaml   # 8-bit, 500+ steps
```

**Implementation**:
1. Create `configs/` directory
2. Define YAML schema for training params (quantization, batch size, steps, etc.)
3. Add `--config` argument to `run_trl_grpo.py`
4. CLI args should override config file values
5. Consider wiring up existing `src/config.py` dataclasses

**Existing config**: `src/config.py` has `HardwareConfig`, `LoRAConfig`, `TrainingConfig` dataclasses but they're not used by the GRPO script.

**Key parameters to expose in profiles**:
- `num_generations` (GRPO group size, currently 4) - larger = better gradient signal, more memory
- `load_in_4bit` vs `load_in_8bit` - quality/memory tradeoff
- `gradient_accumulation_steps` - effective batch size
- `learning_rate` - currently 1e-5
- `early_stopping_patience` - currently 3 evals

---

## Philosophy Reminder

This project implements Lonergan's four cognitive operations:
1. **Attention** (prosoche/shema/dharana) - BE ATTENTIVE
2. **Understanding** (noesis/binah/vijnana) - BE INTELLIGENT
3. **Judgment** (krisis/mishpat/viveka) - BE REASONABLE
4. **Decision** (prohairesis/ratzon/sankalpa) - BE RESPONSIBLE

The virtually unconditioned: A judgment is warranted when conditions are fulfilled.
- Yes = conditions identifiably fulfilled
- No = conditions identifiably NOT fulfilled
- Insufficient = cannot determine fulfillment

Read `docs/lonergan_synopsis.md` if philosophical context is needed.

---

## Successor Agent Prompt: Build the Full Training Pipeline

**Your Task**: Integrate the test architecture modules into the training pipeline.

### Step 1: Analyze Current Data Coverage

**DATA FORMAT NOTE**: Existing data has fields nested in `metadata`, not top-level:
```python
# Example structure:
{
    "messages": [...],
    "metadata": {
        "domain": "mathematical",
        "judgment_type": "Yes",  # This is expected_judgment
        "difficulty": "medium",  # String, not int - needs mapping
        ...
    }
}
```

You'll need to either:
1. Update coverage analyzer to read from metadata, OR
2. Transform data to flat format first

```python
from src.coverage.analyzer import CoverageAnalyzer
import json

# Load and transform existing data
DIFFICULTY_MAP = {"introductory": 1, "easy": 2, "medium": 3, "hard": 4, "expert": 5}

examples = []
with open("data/oracle_generated/judgment_train.jsonl") as f:
    for line in f:
        raw = json.loads(line)
        meta = raw.get("metadata", {})
        # Note: analyzer expects "judgment" (capitalized), not "expected_judgment"
        JUDGMENT_MAP = {"yes": "Yes", "no": "No", "insufficient": "Insufficient"}
        jtype = meta.get("judgment_type", "").lower()
        examples.append({
            "domain": meta.get("domain"),
            "judgment": JUDGMENT_MAP.get(jtype, jtype),  # Must be capitalized
            "difficulty": DIFFICULTY_MAP.get(meta.get("difficulty"), 3),
            "has_distractor": False  # Current data has no distractors
        })

# Analyze coverage
analyzer = CoverageAnalyzer()
report = analyzer.analyze(examples)
print(f"Coverage: {report.coverage_percentage:.1f}%")
print(f"Gaps: {len(report.gaps)}")
```

### Step 2: Generate Contrastive Distractors
```python
from src.generation.distractor_generator import generate_distractors_for_example
# Target: 30% of training data should have contrastive distractors
```

### Step 3: Fill Coverage Gaps
**Current baseline**: 23.3% coverage (35/150 cells filled)
Target: >80% of 150 cells

**Key gaps**:
- ALL distractor cells are empty (has_distractor=True)
- Several domain/judgment combinations have no examples
- Difficulties 1 and 5 are underrepresented

### Step 4: Wire Enhanced Reward
Replace reward function in `scripts/run_trl_grpo.py`:
```python
from src.training.enhanced_reward import reward_fn
```

### Step 5: Create Integration Script
Create `scripts/run_enhanced_training.py` that orchestrates:
1. Load and analyze coverage
2. Generate distractors
3. Fill gaps
4. Run GRPO with enhanced reward

### Success Criteria
- [ ] Coverage increases from 23.3% to >80% of 150 cells
- [ ] 30% of training data has contrastive distractors (fills all distractor=True cells)
- [ ] Enhanced reward function integrated
- [ ] Training runs end-to-end with improved data

### User Notes
- **Gemini Pro 3** should be used for all synthetic test generation (API key in `.env`)
- Research the exact model name if needed

### Key Files to Read First
1. `docs/implementation_plan.md` - Full architecture
2. `src/training/enhanced_reward.py` - New reward function
3. `src/coverage/analyzer.py` - Coverage analysis
4. `src/generation/distractor_generator.py` - Distractor generation
