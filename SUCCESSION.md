# Succession Notes for Next Agent

*Last updated: 2026-01-22 (Session 6)*

## Immediate Context

**PHASE**: Abandon custom pipelined trainer. Return to TRL-based `run_enhanced_training.py`.

**WHAT JUST HAPPENED (Session 6)**:
1. Attempted to debug pipelined trainer generation hang
2. Discovered generation is NOT hanging - it's just **extremely slow** (2 tok/s vs expected 10+ tok/s)
3. Identified that SDPA (Scaled Dot Product Attention) works fine with base model (13.8 tok/s)
4. With LoRA: 8.1 tok/s. With TRL+LoRA: 2 tok/s - something in TRL's loading is disabling efficient attention
5. Could not isolate the root cause in reasonable time
6. **Decision**: Abandon custom pipelined trainer, use TRL's native training which works correctly

**KEY INSIGHT**: The TRL-based `run_enhanced_training.py` script works correctly and at reasonable speed. The custom `pipelined_trainer.py` was an optimization attempt that introduced complexity we couldn't debug efficiently.

## What Works

### Use This: `scripts/run_enhanced_training.py`
- Uses `CognitiveGRPOTrainer` (fixes TRL column preservation bug)
- TRL handles generation internally at proper speed
- Gemini evaluation works (model updated to `gemini-3-flash-preview`)

### Don't Use: `src/training/pipelined_trainer.py`
- Cross-step async pipelining concept is sound
- Implementation has undiagnosed generation slowdown (2 tok/s)
- Not worth debugging when TRL-based approach works

## Fixed This Session

1. **Gemini model name**: Updated from `gemini-2.5-flash-preview-05-20` (404) to `gemini-3-flash-preview`
2. **Logging**: Rewrote `src/evaluation/logging_config.py` to use per-component QueueHandler for non-blocking async-safe logging
3. **Async evaluator order**: Moved async Gemini thread startup to happen before model loading

## V2 Dataset

**Location**: `data/oracle_generated/judgment_v2_train.jsonl` (896 examples)

Split from full dataset:
- `judgment_v2_train.jsonl` - 896 examples (training)
- `judgment_v2_val.jsonl` - 100 examples (validation)

## Quick Commands

```bash
# Activate environment
cd /home/dgk/projects/cognitiveop_attunement
source .venv/bin/activate

# Run enhanced training (THE WORKING APPROACH)
CUDA_VISIBLE_DEVICES=0 python scripts/run_enhanced_training.py \
  --data data/oracle_generated/judgment_v2_train.jsonl \
  --output models/judgment_v2 \
  --steps 20 \
  --batch-size 1 \
  --num-generations 2

# Check Gemini connectivity
python -c "
from google import genai
import os
client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
response = client.models.generate_content(model='gemini-3-flash-preview', contents='Say hello')
print(response.text)
"

# Run tests
pytest tests/ -v --tb=short
```

## Technical Details

### Generation Speed Findings

| Configuration | tok/s | Notes |
|--------------|-------|-------|
| Base model + SDPA | 13.8 | Fast, expected |
| Base + LoRA + SDPA | 8.1 | Acceptable |
| TRL + LoRA (direct load) | ~2.0 | Something disables SDPA |
| TRL native training | Works | TRL handles it internally |

The mystery: TRL's `GRPOTrainer` loading seems to disable efficient attention even when `attn_implementation="sdpa"` is passed in `model_init_kwargs`. But TRL's *internal* generation during training works fine.

### Gemini Model

Working: `gemini-3-flash-preview`
Dead: `gemini-2.5-flash-preview-05-20`

Available models (as of 2026-01-22):
- `gemini-3-flash-preview`
- `gemini-2.5-flash`
- `gemini-2.5-pro`
- `gemini-2.0-flash`

## Files Modified This Session

- `src/training/pipelined_trainer.py` - Setup order, logging (but don't use this file)
- `src/evaluation/logging_config.py` - Rewrote with per-component QueueHandler
- `src/evaluation/async_reward.py` - Added init/ready log messages
- Model names updated to `gemini-3-flash-preview` in multiple files

## Test Files Created (Can Delete)

- `test_gen_minimal.py`
- `test_gen_class.py`
- `test_gen_import.py`
- `test_gen_trainer.py`
- `test_gen_async.py`
- `test_gen_setup.py`
- `test_async_timing.py`
- `test_single_step.py`
- `test_gen_with_async.py`
- `test_gen_real_prompt.py`
- `test_gen_speed.py`
- `test_gen_lora.py`
- `test_gen_trl_nocompile.py`
- `test_gen_real_nocompile.py`

---

## Continuation Prompt for Next Agent

```
# Continuation: Run Enhanced Training with V2 Data

## Context
You are continuing work on the Cognitive Organism project - training Qwen2.5-7B-Instruct
to perform Lonergan's cognitive operation of JUDGMENT using GRPO.

Previous session attempted to optimize training with a custom pipelined trainer but
encountered undiagnosed generation slowdown. Decision was made to abandon that approach
and return to the working TRL-based `run_enhanced_training.py`.

## Your Task

1. **Clean up test files** (optional but recommended):
   ```bash
   rm -f test_gen_*.py test_async_*.py test_single_step.py
   ```

2. **Verify Gemini connectivity**:
   ```bash
   source .venv/bin/activate
   python -c "
   from src.evaluation.llm_evaluator import LlmEvaluator
   eval = LlmEvaluator(provider='gemini', model_name='gemini-3-flash-preview', use_cache=False)
   print('Gemini OK')
   "
   ```

3. **Run short validation training** (10-20 steps):
   ```bash
   CUDA_VISIBLE_DEVICES=0 python scripts/run_enhanced_training.py \
     --data data/oracle_generated/judgment_v2_train.jsonl \
     --output models/v2_validation \
     --steps 20 \
     --batch-size 1 \
     --num-generations 2 \
     --clean
   ```

4. **Monitor training**:
   - Check `logs/training_*.log` for training progress
   - Check `logs/gemini_*.log` for evaluation details
   - Verify reward variance > 0 (if all rewards identical, training won't learn)
   - Verify loss is decreasing or reward is increasing

5. **Document results** in SUCCESSION.md for next agent

## Key Files

- `scripts/run_enhanced_training.py` - Main training script (USE THIS)
- `src/training/cognitive_grpo_trainer.py` - CognitiveGRPOTrainer (preserves custom columns)
- `data/oracle_generated/judgment_v2_train.jsonl` - Training data (896 examples)

## Success Criteria
- [ ] Training runs without errors for 20 steps
- [ ] Reward variance > 0 (model produces diverse outputs)
- [ ] Learning signal visible (loss trending down OR mean reward trending up)
- [ ] Results documented

## If Training Fails

Common issues:
1. **Gemini 404**: Model name changed. Check available models with:
   ```python
   from google import genai
   import os
   client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
   for m in client.models.list():
       if 'flash' in m.name.lower():
           print(m.name)
   ```

2. **CUDA OOM**: Reduce batch size or num_generations

3. **Zero reward variance**: All completions getting same reward - check reward function logic

4. **Column dropped**: Use CognitiveGRPOTrainer, not base GRPOTrainer
```
