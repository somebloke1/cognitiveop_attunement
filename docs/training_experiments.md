# Training Experiments Log

## Experiment 1: GRPO Pipeline Validation (2026-01-19)

### Configuration
- **Model**: Qwen/Qwen2.5-7B-Instruct (4-bit quantized)
- **Method**: TRL GRPOTrainer with LoRA
- **LoRA Config**: r=64, alpha=128, dropout=0.05
- **Target Modules**: q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, gate_proj
- **Dataset**: 405 training examples (judgment_train.jsonl)
- **Steps**: 20 (validation run)
- **Batch Size**: 1
- **Generations per Prompt**: 4

### Results
| Metric | Start | End | Change |
|--------|-------|-----|--------|
| Loss | 0.0156 | 0.0095 | -39% |
| Mean Reward | 0.0275 | 0.035 | +27% |
| Completion Length | 332 | 198 | -40% |
| Training Time | - | 19 min | - |

### Observations
- Model learns to produce shorter, more focused responses
- Judgment structure (Yes/No/Insufficient) emerging
- Condition fulfillment language appearing in outputs
- No OOM issues on RTX 3090

### Sample Output (Post-Training)
```
Affirm (Yes)

Explanation:
1. The conditioned proposition is that water boils at 100°C under standard pressure.
2. The link between the proposition and its conditions...
3. The evidence confirms... conditions are fulfilled.
```

---

## Experiment 2: Extended Training (PENDING)

### Planned Configuration
- **Steps**: 100-200
- **Expected Duration**: ~2-3 hours
- **Checkpoint Interval**: Every 50 steps

### Hypotheses
- Loss should continue decreasing
- Reward should stabilize around 0.5+
- Judgment accuracy on validation should improve

---

## Reward Function Design

The reward function (`judgment_reward_fn`) in `scripts/run_trl_grpo.py` implements:

### Positive Rewards
| Condition | Reward | Rationale |
|-----------|--------|-----------|
| Contains "Judgment: Yes" | +0.3 | Explicit judgment declaration |
| Contains "Judgment: No" | +0.3 | Explicit judgment declaration |
| Contains "Judgment: Insufficient" | +0.3 | Explicit judgment declaration |
| Contains "fulfilled" or "conditions" | +0.2 | Discusses condition fulfillment |
| Contains "because/since/therefore" | +0.2 | Provides reasoning |

### Penalties
| Condition | Penalty | Rationale |
|-----------|---------|-----------|
| Output < 50 chars | -0.2 | Too short to be substantive |
| Missing "judgment:" | -0.3 | Lacks required structure |

### Future Improvements
1. Add oracle verification for judgment correctness
2. Domain-specific reward components
3. Penalize counterposition patterns
4. Reward proper virtually-unconditioned structure

---

## Data Distribution Analysis

### By Domain (450 total)
| Domain | Count | % |
|--------|-------|---|
| Mathematical | 106 | 24% |
| Empirical | 106 | 24% |
| Pop Science | 96 | 21% |
| Philosophic | 76 | 17% |
| Common Sense | 66 | 15% |

### By Judgment Type
| Type | Count | % |
|------|-------|---|
| No | 200 | 44% |
| Insufficient | 90 | 20% |
| Yes | 80 | 18% |
| (Level-themed) | 80 | 18% |

### Notes
- "No" judgments overrepresented (expected: pop science domain mostly produces "No")
- Some difficult cells failed verification (common_sense/hard/Yes, philosophic/medium/No)
- This is by design - verifier correctly rejects weak cases
