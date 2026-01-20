# Improved Reward Function Proposal

## Problem

The current reward function in `scripts/run_trl_grpo.py` only checks for structural markers (presence of "judgment: yes", "conditions", etc.) but ignores the oracle-verified expected judgments in the training data.

This trains the model to produce judgment-shaped text, not correct judgments.

## Root Cause

1. Dataset creation discards ground truth:
```python
# Current (line 235)
dataset = Dataset.from_list([{"prompt": ex["prompt"]} for ex in examples])
```

2. Reward function has no access to expected outputs:
```python
# Current signature
def judgment_reward_fn(prompts: list[str], completions: list[str], **kwargs)
```

## Solution

TRL's GRPOTrainer passes extra dataset columns directly to reward functions as kwargs.

### Change 1: Dataset Creation (line 235)

```python
# BEFORE
dataset = Dataset.from_list([{"prompt": ex["prompt"]} for ex in examples])

# AFTER
dataset = Dataset.from_list([{
    "prompt": ex["prompt"],
    "expected_judgment": ex["judgment_type"],  # "Yes", "No", "Insufficient"
    "domain": ex["domain"],
    "expected_output": ex["expected"],  # Full expected response for optional similarity
} for ex in examples])
```

### Change 2: Validation Dataset (line 242)

```python
# BEFORE
val_dataset = Dataset.from_list([{"prompt": ex["prompt"]} for ex in val_examples])

# AFTER
val_dataset = Dataset.from_list([{
    "prompt": ex["prompt"],
    "expected_judgment": ex["judgment_type"],
    "domain": ex["domain"],
    "expected_output": ex["expected"],
} for ex in val_examples])
```

### Change 3: Reward Function

```python
def judgment_reward_fn(
    completions: list[str],
    expected_judgment: list[str],
    domain: list[str] = None,
    **kwargs
) -> list[float]:
    """
    Reward function for judgment quality.

    Rewards:
    - CORRECTNESS (0.5): Does the judgment match the oracle-verified answer?
    - STRUCTURE (0.2): Does it have proper judgment markers?
    - REASONING (0.2): Does it explain conditions/fulfillment?
    - DOMAIN APPROPRIATENESS (0.1): Does reasoning fit the domain?

    Penalties:
    - Missing judgment marker (-0.3)
    - Very short response (-0.2)
    """
    rewards = []

    for i, completion in enumerate(completions):
        output_lower = completion.lower()
        reward = 0.0

        # === CORRECTNESS (most important) ===
        # Extract what the model actually judged
        model_judgment = None
        if "judgment: yes" in output_lower:
            model_judgment = "Yes"
        elif "judgment: no" in output_lower:
            model_judgment = "No"
        elif "judgment: insufficient" in output_lower:
            model_judgment = "Insufficient"

        # Compare to ground truth
        expected = expected_judgment[i] if expected_judgment else None
        if model_judgment and expected:
            if model_judgment == expected:
                reward += 0.5  # Correct judgment
            else:
                reward -= 0.2  # Wrong judgment (mild penalty, still learning structure)

        # === STRUCTURE ===
        if model_judgment is not None:
            reward += 0.2  # Has explicit judgment
        else:
            reward -= 0.3  # Missing judgment marker

        # === REASONING ===
        # Check for Lonergan-appropriate structure
        has_conditions = "condition" in output_lower
        has_fulfillment = "fulfill" in output_lower or "met" in output_lower
        has_link = "link" in output_lower or "if " in output_lower
        has_causal = any(w in output_lower for w in ["because", "since", "therefore", "thus"])

        reasoning_score = sum([has_conditions, has_fulfillment, has_link, has_causal])
        reward += min(0.2, reasoning_score * 0.05)  # Up to 0.2 for reasoning

        # === DOMAIN APPROPRIATENESS (optional bonus) ===
        if domain and domain[i]:
            d = domain[i].lower()
            if d == "mathematical" and ("proof" in output_lower or "axiom" in output_lower or "theorem" in output_lower):
                reward += 0.1
            elif d == "empirical" and ("evidence" in output_lower or "data" in output_lower or "observation" in output_lower):
                reward += 0.1
            elif d == "common_sense" and ("practical" in output_lower or "situation" in output_lower):
                reward += 0.1
            elif d == "pop_science" and ("unfulfilled" in output_lower or "claimed" in output_lower or "authority" in output_lower):
                reward += 0.1
            elif d == "philosophic" and ("self" in output_lower or "coherent" in output_lower or "performative" in output_lower):
                reward += 0.1

        # === PENALTIES ===
        if len(completion) < 50:
            reward -= 0.2  # Too short

        rewards.append(reward)

    return rewards
```

## Reward Score Breakdown

| Component | Max Score | Description |
|-----------|-----------|-------------|
| Correctness | +0.5 / -0.2 | Matches oracle-verified judgment |
| Structure | +0.2 / -0.3 | Has explicit "Judgment: X" marker |
| Reasoning | +0.2 | Conditions, fulfillment, causal language |
| Domain | +0.1 | Domain-appropriate vocabulary |
| Length | -0.2 | Penalty if < 50 chars |

**Max possible reward**: 1.0
**Min possible reward**: -0.7 (wrong judgment, no structure, too short)

## Philosophical Grounding

This reward structure maps to Lonergan's virtually unconditioned:

1. **Correctness** = The judgment IS or IS NOT fulfilled (verified by oracle)
2. **Structure** = The form of judgment (explicit commitment: Yes/No/Insufficient)
3. **Reasoning** = The link between conditions and fulfillment
4. **Domain** = Appropriate type of evidence for the domain (mathematical necessity vs empirical verification vs practical sufficiency)

We're not just training form - we're training the model to reach the correct judgment through proper cognitive operation.

## Questions for Review

1. Should wrong judgments have a stronger penalty (-0.2 seems mild)?
2. Should we add semantic similarity to expected_output for reasoning quality?
3. Is the domain bonus worth the added complexity?
4. Should "Insufficient" get special treatment (often harder to judge)?
