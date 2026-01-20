# Succession Notes for Next Agent

*Last updated: 2026-01-20*

## Immediate Context

**IMPORTANT**: The reward function was significantly improved on 2026-01-20. Future training runs should use the updated `scripts/run_trl_grpo.py`.

## Project State Summary

The Cognitive Organism training pipeline has a **corrected reward function**:

1. **Oracle Pipeline**: Gemini-based data generation with multi-tier verification working
2. **Dataset**: 450 verified examples ready (405 train, 45 val)
3. **GRPO Training**: TRL-based pipeline with **correctness-dominant reward function**
4. **Key Fix**: Reward function now uses expected_judgment from training data, not just structural markers

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
| `data/oracle_generated/judgment_train.jsonl` | Training data |
| `models/judgment_grpo_test/` | First successful adapter |
| `docs/training_experiments.md` | Training log |
| `CLAUDE.md` | Project state and philosophy |

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
