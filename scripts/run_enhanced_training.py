#!/usr/bin/env python3
"""
Enhanced Training Pipeline Orchestrator

This script integrates:
1. Coverage analysis (src/coverage/analyzer.py)
2. Distractor generation (src/generation/distractor_generator.py)
3. Gap filling via Gemini oracle
4. Enhanced reward function (src/training/enhanced_reward.py)
5. TRL GRPO training

Usage:
    # Analyze only
    python scripts/run_enhanced_training.py --analyze-only

    # Generate distractors and fill gaps (no training)
    python scripts/run_enhanced_training.py --generate-only

    # Full pipeline: generate + train
    python scripts/run_enhanced_training.py --steps 100 --output models/judgment_enhanced

    # Run training with existing enhanced data
    python scripts/run_enhanced_training.py --skip-generation --steps 100
"""

import argparse
import json
import os
import sys
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# Data Transformation Constants
# =============================================================================

DIFFICULTY_MAP = {
    "introductory": 1,
    "easy": 2,
    "medium": 3,
    "hard": 4,
    "expert": 5,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
}

DIFFICULTY_MAP_REVERSE = {
    1: "introductory",
    2: "easy",
    3: "medium",
    4: "hard",
    5: "expert",
}

JUDGMENT_MAP = {
    "yes": "Yes",
    "no": "No",
    "insufficient": "Insufficient",
    "Yes": "Yes",
    "No": "No",
    "Insufficient": "Insufficient",
}


# =============================================================================
# Step 1: Coverage Analysis
# =============================================================================


def analyze_coverage(data_path: Path, min_count: int = 2) -> Dict[str, Any]:
    """Analyze coverage of existing data."""
    from src.coverage.analyzer import (
        build_coverage_matrix,
        find_coverage_gaps,
        get_coverage_percentage,
    )

    # Load and transform data
    examples = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                raw = json.loads(line)
                meta = raw.get("metadata", {})
                examples.append(
                    {
                        "domain": meta.get("domain", "empirical"),
                        "judgment": JUDGMENT_MAP.get(
                            meta.get("judgment_type", ""), "Yes"
                        ),
                        "difficulty": DIFFICULTY_MAP.get(meta.get("difficulty", 3), 3),
                        "has_distractor": meta.get("has_distractor", False),
                    }
                )

    matrix = build_coverage_matrix(examples)
    gaps = find_coverage_gaps(matrix, min_count)
    coverage_pct = get_coverage_percentage(matrix)

    return {
        "examples": examples,
        "matrix": matrix,
        "gaps": gaps,
        "coverage_percentage": coverage_pct,
        "total_cells": len(matrix),
        "filled_cells": sum(1 for c in matrix.values() if c > 0),
    }


# =============================================================================
# Step 2: Generate Distractors from Existing Examples
# =============================================================================


def generate_distractors_batch(
    data_path: Path, output_path: Path, target_ratio: float = 0.3
):
    """Generate distractors for a fraction of existing examples."""
    from src.generation.distractor_generator import (
        DistractorGenerator,
        JudgmentSample,
        DistractorType,
    )

    print(f"\n{'=' * 60}")
    print("STEP 2: GENERATING DISTRACTORS")
    print(f"{'=' * 60}")

    # Load existing data
    examples = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    # Calculate how many distractors to generate
    num_distractors = int(len(examples) * target_ratio)
    print(f"Existing examples: {len(examples)}")
    print(f"Target distractors: {num_distractors} ({target_ratio * 100:.0f}%)")

    # Select examples to create distractors from
    selected = random.sample(examples, min(num_distractors, len(examples)))

    generator = DistractorGenerator()
    distractors = []
    distractor_types = [
        DistractorType.MISALIGNED_PHASE,
        DistractorType.INVERTED_JUDGMENT,
        DistractorType.REASONING_ERROR,
        DistractorType.EVIDENCE_HALLUCINATION,
    ]

    for i, ex in enumerate(selected):
        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{num_distractors} distractors...")

        # Extract sample data
        msgs = ex.get("messages", [])
        meta = ex.get("metadata", {})

        # Find user and assistant messages
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
        asst_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")

        # Parse out proposition, evidence, conditions from user message
        proposition = ""
        evidence = ""
        conditions = []

        lines = user_msg.split("\n")
        for j, line in enumerate(lines):
            if line.startswith("Proposition:"):
                proposition = line.replace("Proposition:", "").strip()
            elif line.startswith("Evidence:"):
                evidence = line.replace("Evidence:", "").strip()
            elif line.startswith("Conditions for truth:"):
                # Collect conditions from following lines
                for k in range(j + 1, len(lines)):
                    cond_line = lines[k].strip()
                    if cond_line.startswith("-"):
                        conditions.append(cond_line[1:].strip())
                    elif cond_line == "":
                        continue
                    elif not cond_line.startswith("-"):
                        break

        # Create JudgmentSample
        sample = JudgmentSample(
            proposition=proposition,
            evidence=evidence,
            conditions=conditions,
            conditions_fulfilled={c: True for c in conditions},  # Placeholder
            judgment=JUDGMENT_MAP.get(meta.get("judgment_type", ""), "Yes"),
            judgment_reasoning=asst_msg,
            domain=meta.get("domain", "empirical"),
            difficulty=DIFFICULTY_MAP.get(meta.get("difficulty", 3), 3),
        )

        # Generate distractor
        dtype = distractor_types[i % len(distractor_types)]
        try:
            if dtype == DistractorType.MISALIGNED_PHASE:
                distractor = generator.generate_misaligned_phase(sample)
            elif dtype == DistractorType.INVERTED_JUDGMENT:
                distractor = generator.generate_inverted_judgment(sample)
            elif dtype == DistractorType.REASONING_ERROR:
                distractor = generator.generate_reasoning_error(sample)
            else:
                distractor = generator.generate_evidence_hallucination(sample)

            # Convert to training format
            distractor_ex = {
                "messages": [
                    msgs[0],  # Keep system prompt
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": distractor.distractor_completion},
                ],
                "metadata": {
                    "level": "judgment",
                    "domain": sample.domain,
                    "judgment_type": meta.get("judgment_type", "Yes"),
                    "difficulty": DIFFICULTY_MAP_REVERSE.get(
                        sample.difficulty, "medium"
                    ),
                    "has_distractor": True,
                    "distractor_type": dtype.value,
                    "generated_by": "distractor_generator",
                },
            }
            distractors.append(distractor_ex)

        except Exception as e:
            print(f"  Warning: Failed to generate distractor for example {i}: {e}")

    # Save distractors
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for d in distractors:
            f.write(json.dumps(d) + "\n")

    print(f"Generated {len(distractors)} distractors -> {output_path}")
    return distractors


# =============================================================================
# Step 3: Fill Coverage Gaps with Gemini Oracle
# =============================================================================


def fill_coverage_gaps(
    gaps: List[Any], output_path: Path, examples_per_gap: int = 2, max_gaps: int = 50
):
    """Fill coverage gaps using Gemini oracle."""
    import dspy
    from src.dspy_cognitive.data_generation import (
        GenerateScenarioSignature,
        GenerateJudgmentSignature,
    )
    from src.dspy_cognitive.lonergan_context import get_judgment_context

    print(f"\n{'=' * 60}")
    print("STEP 3: FILLING COVERAGE GAPS")
    print(f"{'=' * 60}")

    # Initialize DSPy with Gemini
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found in environment")
        return []

    lm = dspy.LM("gemini/gemini-2.0-flash", api_key=api_key)
    dspy.configure(lm=lm)

    scenario_gen = dspy.Predict(GenerateScenarioSignature)
    judgment_gen = dspy.Predict(GenerateJudgmentSignature)

    # Sort gaps by priority (most empty first)
    sorted_gaps = sorted(gaps, key=lambda g: (g.current_count, g.priority))[:max_gaps]

    print(
        f"Filling {len(sorted_gaps)} priority gaps with {examples_per_gap} examples each"
    )

    generated = []
    for i, gap in enumerate(sorted_gaps):
        if (i + 1) % 10 == 0:
            print(f"  Processing gap {i + 1}/{len(sorted_gaps)}...")

        for _ in range(examples_per_gap):
            try:
                # Generate scenario
                scenario = scenario_gen(
                    domain=gap.domain,
                    difficulty=DIFFICULTY_MAP_REVERSE.get(gap.difficulty, "medium"),
                    expected_judgment=gap.judgment,
                )

                # Generate judgment
                judgment = judgment_gen(
                    lonergan_context=get_judgment_context(gap.domain),
                    proposition=scenario.proposition,
                    evidence=scenario.evidence,
                    conditions=scenario.conditions_for_truth,
                    domain=gap.domain,
                )

                # Build training example
                system_prompt = """You are a Judgment cognitive operation (krisis/mishpat/viveka). Evaluate whether a proposition is true by grasping the virtually unconditioned.

If conditions are fulfilled, affirm (Yes). If identifiably not fulfilled, deny (No). If cannot determine, withhold (Insufficient).

OUTPUT FORMAT (follow exactly):
JUDGMENT: Yes, No, or Insufficient
REASONING: Your explanation of condition fulfillment

Example:
JUDGMENT: Insufficient
REASONING: The first condition requires X, but the evidence only shows Y. The second condition...

Be reasonable (episteme, not doxa). Demand sufficient reason."""

                conditions_text = "\n".join(
                    f"- {c}" for c in scenario.conditions_for_truth
                )
                user_content = f"""Proposition: {scenario.proposition}

Evidence: {scenario.evidence}

Conditions for truth:
{conditions_text}"""

                fulfillment_text = ""
                for j, (cond, status) in enumerate(
                    zip(scenario.conditions_for_truth, judgment.condition_fulfillment)
                ):
                    fulfillment_text += f"- {cond}: {status}\n"

                assistant_content = f"""Condition fulfillment:
{fulfillment_text}
Judgment: {judgment.judgment}

Reasoning: {judgment.reasoning}"""

                example = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content},
                    ],
                    "metadata": {
                        "level": "judgment",
                        "domain": gap.domain,
                        "judgment_type": judgment.judgment,
                        "difficulty": DIFFICULTY_MAP_REVERSE.get(
                            gap.difficulty, "medium"
                        ),
                        "has_distractor": gap.has_distractor,
                        "generated_by": "gap_filler",
                    },
                }
                generated.append(example)

            except Exception as e:
                print(
                    f"  Warning: Failed to generate for gap {gap.domain}/{gap.judgment}/D={gap.difficulty}: {e}"
                )

    # Save generated examples
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in generated:
            f.write(json.dumps(ex) + "\n")

    print(f"Generated {len(generated)} gap-filling examples -> {output_path}")
    return generated


# =============================================================================
# Step 4: Combine Enhanced Dataset
# =============================================================================


def combine_datasets(
    original_path: Path, distractors_path: Path, gap_fill_path: Path, output_path: Path
):
    """Combine original, distractor, and gap-fill datasets."""
    print(f"\n{'=' * 60}")
    print("STEP 4: COMBINING DATASETS")
    print(f"{'=' * 60}")

    combined = []

    # Load original
    if original_path.exists():
        with open(original_path) as f:
            for line in f:
                if line.strip():
                    combined.append(json.loads(line))
        print(f"Original examples: {len(combined)}")

    # Load distractors
    distractor_count = 0
    if distractors_path.exists():
        with open(distractors_path) as f:
            for line in f:
                if line.strip():
                    combined.append(json.loads(line))
                    distractor_count += 1
        print(f"Distractor examples: {distractor_count}")

    # Load gap fillers
    gap_count = 0
    if gap_fill_path.exists():
        with open(gap_fill_path) as f:
            for line in f:
                if line.strip():
                    combined.append(json.loads(line))
                    gap_count += 1
        print(f"Gap-fill examples: {gap_count}")

    # Shuffle
    random.shuffle(combined)

    # Save combined
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in combined:
            f.write(json.dumps(ex) + "\n")

    print(f"Total combined: {len(combined)} -> {output_path}")
    return combined


# =============================================================================
# Step 5: Training with Enhanced Reward
# =============================================================================


def run_training(
    data_path: Path,
    output_path: Path,
    steps: int = 100,
    batch_size: int = 1,
    num_generations: int = 5,
    learning_rate: float = 5e-5,
    warmup_ratio: float = 0.1,
    lr_scheduler: str = "cosine",
    save_steps: int = 500,
    eval_steps: int = 100,
    use_batched_eval: bool = True,
    use_cache: bool = True,
    cache_ttl: int = 172800,
    use_async: bool = False,
    use_pipelined_v2: bool = False,
):
    """Run GRPO training with enhanced reward function."""
    print(f"\n{'=' * 60}")
    print("STEP 5: RUNNING ENHANCED GRPO TRAINING")
    print(f"{'=' * 60}")
    print(f"Data: {data_path}")
    print(f"Output: {output_path}")
    print(f"Steps: {steps}")
    print(f"Batch size: {batch_size}")
    print(f"Num generations: {num_generations}")
    print(f"Learning rate: {learning_rate}")
    print(f"Warmup ratio: {warmup_ratio} ({int(steps * warmup_ratio)} steps)")
    print(f"LR scheduler: {lr_scheduler}")
    print(f"Save steps: {save_steps}")
    print(f"Eval steps: {eval_steps}")
    print(f"Batched LLM evaluation: {use_batched_eval}")
    print(f"Gemini context caching: {use_cache}" + (f" (TTL: {cache_ttl}s)" if use_cache else ""))
    print(f"Async Gemini evaluation: {use_async}")
    print(f"PipelinedTrainerV2: {use_pipelined_v2}")
    print(f"Logging: tensorboard -> {output_path}/logs")
    
    # If using V2 pipelined trainer, delegate to it entirely
    if use_pipelined_v2:
        print("\n*** Using PipelinedTrainerV2 with TRL batched generation ***")
        return train_pipelined_v2(
            data_path=str(data_path),
            output_dir=str(output_path),
            num_steps=steps,
            batch_size=batch_size,
            num_generations=num_generations,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            use_gemini_cache=use_cache,
            gemini_cache_ttl=cache_ttl,
            save_steps=save_steps,
        )

    # CRITICAL: Only use RTX 3090 (device 0), not the 1660 Super
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Limit torch compile workers to reduce RAM usage (default 16 is excessive)
    os.environ["TORCH_COMPILE_MAX_WORKERS"] = "4"

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model
    from trl import GRPOConfig

    # Import our custom trainer that preserves cognitive columns
    from src.training.cognitive_grpo_trainer import CognitiveGRPOTrainer
    from src.training.pipelined_grpo_trainer import PipelinedGRPOTrainer
    from src.training.pipelined_trainer_v2 import train_pipelined_v2

    # Import our hybrid reward function (LLM-based semantic evaluation)
    from src.evaluation.hybrid_reward import (
        hybrid_reward_fn,
        get_hybrid_reward_instance,
        HybridRewardConfig,
    )
    from src.evaluation.async_reward import PipelinedHybridReward
    from src.evaluation.llm_evaluator import LlmEvaluator

    # Configure reward function
    reward_config = HybridRewardConfig(
        use_batched_evaluation=use_batched_eval,
        use_gemini_cache=use_cache,
        gemini_cache_ttl=cache_ttl,
    )
    reward_instance = get_hybrid_reward_instance(reward_config)
    
    # Initialize cache before training (creates Gemini cache if enabled)
    reward_instance.initialize()

    # Load data
    examples = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    print(f"Training examples: {len(examples)}")

    # Convert to dataset format
    def extract_fields(ex):
        msgs = ex.get("messages", [])
        meta = ex.get("metadata", {})

        system = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        assistant = next((m["content"] for m in msgs if m["role"] == "assistant"), "")

        prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"

        # Parse proposition, evidence, conditions from user message
        # Expected format:
        #   Proposition: ...
        #   Evidence: ...
        #   Conditions for truth:
        #   - condition 1
        #   - condition 2
        #
        #   Temporal context:
        #   - Evidence timeframe: ...
        #   - Aspectual status: ...
        proposition = ""
        evidence = ""
        conditions = ""

        # Section headers that end the conditions section
        section_headers = (
            "Proposition:",
            "Evidence:",
            "Conditions for truth:",
            "Temporal context:",
        )

        lines = user.split("\n")
        current_section = None
        condition_lines = []

        for line in lines:
            line_stripped = line.strip()
            if line_stripped.startswith("Proposition:"):
                current_section = "proposition"
                proposition = line_stripped.replace("Proposition:", "").strip()
            elif line_stripped.startswith("Evidence:"):
                current_section = "evidence"
                evidence = line_stripped.replace("Evidence:", "").strip()
            elif line_stripped.startswith("Conditions for truth:"):
                current_section = "conditions"
            elif line_stripped.startswith("Temporal context:"):
                current_section = "temporal"  # Stop collecting conditions
            elif current_section == "conditions" and line_stripped.startswith("-"):
                condition_lines.append(line_stripped)
            elif (
                current_section == "evidence"
                and line_stripped
                and not line_stripped.startswith(section_headers)
            ):
                evidence += " " + line_stripped

        conditions = "\n".join(condition_lines)

        # Extract temporal context section if present
        temporal_context = ""
        temporal_lines = []
        in_temporal = False
        for line in lines:
            line_stripped = line.strip()
            if line_stripped.startswith("Temporal context:"):
                in_temporal = True
                continue
            elif in_temporal:
                if line_stripped.startswith("-"):
                    temporal_lines.append(line_stripped)
                elif line_stripped and not line_stripped.startswith("-"):
                    break  # End of temporal section
        temporal_context = "\n".join(temporal_lines)

        # For clean (post-processed) data, oracle_conditions and oracle_temporal_context
        # are in metadata, not in the user message. Fall back to parsed conditions for
        # backward compatibility with old format.
        oracle_conditions = meta.get("oracle_conditions", conditions)
        oracle_temporal_context = meta.get("oracle_temporal_context", temporal_context)

        return {
            "prompt": prompt,
            "expected_judgment": JUDGMENT_MAP.get(meta.get("expected_judgment", meta.get("judgment_type", "")), "Yes"),
            "domain": meta.get("domain", "empirical"),
            "input_text": user,
            "proposition": proposition,
            "evidence": evidence,
            # Oracle answer keys for evaluator grading
            "oracle_conditions": oracle_conditions,
            "oracle_temporal_context": oracle_temporal_context,
            # Legacy field for backward compatibility
            "conditions": conditions,
            # New fields for enhanced evaluation
            "mode": meta.get("mode", ""),  # formal/empirical/common_sense/dialectical
            "difficulty": meta.get("difficulty", "medium"),
            "temporal_focus": meta.get("temporal_focus", ""),
            "temporal_context": temporal_context,  # Extracted from user message
            "edge_case": meta.get("edge_case", ""),
            "adversarial_context": meta.get("adversarial_context", ""),
            "scenario_notes": meta.get("scenario_notes", ""),  # Oracle's reasoning
        }

    processed = [extract_fields(ex) for ex in examples]
    dataset = Dataset.from_list(processed)

    # Split into train/val
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    val_dataset = split["test"]

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Load model
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # Apply LoRA
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training config
    # num_generations: completions per prompt for GRPO contrastive learning
    # TRL requires batch_size to be divisible by num_generations
    # For speed: higher batch_size = faster generation (4x at batch=4 vs batch=1)
    
    # Ensure batch_size is divisible by num_generations
    if batch_size % num_generations != 0:
        # Round up to nearest multiple
        old_batch = batch_size
        batch_size = ((batch_size // num_generations) + 1) * num_generations
        print(f"Adjusted batch_size from {old_batch} to {batch_size} for divisibility with num_generations={num_generations}")
    
    gradient_accumulation_steps = 1  # Keep simple, batch_size handles throughput

    # Logging directory for tensorboard
    logging_dir = output_path / "logs"
    logging_dir.mkdir(parents=True, exist_ok=True)

    # Eval batch size must be divisible by num_generations
    # Set to num_generations to ensure divisibility
    eval_batch_size = num_generations
    
    training_config = GRPOConfig(
        output_dir=str(output_path),
        num_train_epochs=1,
        max_steps=steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler,
        logging_steps=10,
        logging_dir=str(logging_dir),
        eval_steps=eval_steps,
        eval_strategy="steps",
        save_steps=save_steps,
        save_total_limit=4,
        remove_unused_columns=False,
        num_generations=num_generations,
        max_completion_length=768,
        temperature=0.8,
        report_to="tensorboard",
        # Additional metrics logging
        include_num_input_tokens_seen=False,  # Disabled: PEFT model lacks main_input_name
        log_completions=True,
        gradient_checkpointing=True,  # Critical for memory saving
    )

    # Create trainer based on async flag
    if use_async:
        print("\n*** Using PipelinedGRPOTrainer with async Gemini evaluation ***")
        
        # Create LlmEvaluator with caching (shared config)
        llm_evaluator = LlmEvaluator(
            provider="gemini",
            model_name="gemini-3-flash-preview",
            use_cache=use_cache,
            cache_ttl=cache_ttl,
        )
        
        # Create cache before starting async evaluator
        if use_cache:
            cache_name = llm_evaluator.create_cache()
            if cache_name:
                print(f"Gemini cache created: {cache_name}")
            else:
                print("Warning: Failed to create Gemini cache")
        
        # Create pipelined evaluator
        pipelined_evaluator = PipelinedHybridReward(
            llm_evaluator=llm_evaluator,
            correctness_weight=reward_config.correctness_weight,
            semantic_weight=reward_config.semantic_weight,
        )
        
        trainer = PipelinedGRPOTrainer(
            model=model,
            args=training_config,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
            reward_funcs=hybrid_reward_fn,  # fallback
            pipelined_evaluator=pipelined_evaluator,
        )
    else:
        # Trainer - use CognitiveGRPOTrainer to preserve custom columns
        # (The base GRPOTrainer drops expected_judgment, domain, etc. causing reward collapse)
        trainer = CognitiveGRPOTrainer(
            model=model,
            args=training_config,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
            reward_funcs=hybrid_reward_fn,
        )
        llm_evaluator = None  # For cleanup logic

    # Train with cache cleanup in finally block
    print("\nStarting training...")
    try:
        trainer.train()

        # Save
        trainer.save_model(str(output_path))
        print(f"\nModel saved to: {output_path}")
    finally:
        # Cleanup Gemini cache (if created)
        if use_async and llm_evaluator and llm_evaluator.cache_active:
            llm_evaluator.delete_cache()
            print("Gemini cache deleted")
        else:
            reward_instance.cleanup()

    return trainer


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Enhanced training pipeline")
    parser.add_argument(
        "--data", type=Path, default=Path("data/oracle_generated/judgment_train.jsonl")
    )
    parser.add_argument("--output", type=Path, default=Path("models/judgment_enhanced"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=5, help="Per-device batch size. Higher values (4-8) give 4x faster generation.")
    parser.add_argument(
        "--num-generations",
        type=int,
        default=5,
        help="Number of completions per prompt for GRPO (default: 5)",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=5e-5, help="Peak learning rate"
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Fraction of steps for linear warmup (0.0-1.0)",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "linear", "constant"],
        help="Learning rate scheduler after warmup",
    )
    parser.add_argument(
        "--save-steps", type=int, default=500, help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--eval-steps", type=int, default=100, help="Run validation every N steps"
    )
    parser.add_argument(
        "--no-batch-eval",
        action="store_true",
        help="Disable batched LLM evaluation (use individual parallel calls instead)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable Gemini context caching (caching reduces token costs by ~50%%)",
    )
    parser.add_argument(
        "--cache-ttl",
        type=int,
        default=172800,
        help="Cache TTL in seconds (default: 172800 = 48 hours)",
    )
    parser.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        help="Use async Gemini evaluation (background thread, non-blocking API calls)",
    )
    parser.add_argument(
        "--pipelined-v2",
        dest="use_pipelined_v2",
        action="store_true",
        help="Use PipelinedTrainerV2 with TRL batched generation and true cross-step pipelining",
    )
    parser.add_argument(
        "--analyze-only", action="store_true", help="Only analyze coverage"
    )
    parser.add_argument(
        "--generate-only", action="store_true", help="Generate data but don't train"
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip data generation, use existing enhanced data",
    )
    parser.add_argument("--distractor-ratio", type=float, default=0.3)
    parser.add_argument("--examples-per-gap", type=int, default=2)
    parser.add_argument("--max-gaps", type=int, default=50)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing adapter in output directory before training",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["error", "warn", "info", "debug"],
        default="info",
        help="Logging level: error, warn, info, debug (default: info)",
    )
    args = parser.parse_args()

    # Initialize logging with specified level
    from src.evaluation.logging_config import setup_logging

    setup_logging(level=args.log_level)

    # Handle --clean flag
    if args.clean and args.output.exists():
        import shutil

        print(f"Removing existing adapter: {args.output}")
        shutil.rmtree(args.output)
        print("Done.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Paths
    distractors_path = Path(f"data/enhanced/distractors_{timestamp}.jsonl")
    gap_fill_path = Path(f"data/enhanced/gap_fill_{timestamp}.jsonl")
    combined_path = Path("data/enhanced/judgment_train_enhanced.jsonl")

    # Step 1: Analyze coverage
    print(f"\n{'=' * 60}")
    print("STEP 1: COVERAGE ANALYSIS")
    print(f"{'=' * 60}")

    analysis = analyze_coverage(args.data)
    print(f"Current coverage: {analysis['coverage_percentage']:.1f}%")
    print(f"Filled cells: {analysis['filled_cells']}/{analysis['total_cells']}")
    print(f"Coverage gaps: {len(analysis['gaps'])}")

    if args.analyze_only:
        return

    if not args.skip_generation:
        # Step 2: Generate distractors
        distractors = generate_distractors_batch(
            args.data, distractors_path, args.distractor_ratio
        )

        # Step 3: Fill gaps
        filled = fill_coverage_gaps(
            analysis["gaps"], gap_fill_path, args.examples_per_gap, args.max_gaps
        )

        # Step 4: Combine datasets
        combine_datasets(args.data, distractors_path, gap_fill_path, combined_path)

        # Re-analyze
        print(f"\n{'=' * 60}")
        print("POST-GENERATION COVERAGE ANALYSIS")
        print(f"{'=' * 60}")
        new_analysis = analyze_coverage(combined_path)
        print(f"New coverage: {new_analysis['coverage_percentage']:.1f}%")
        print(
            f"Filled cells: {new_analysis['filled_cells']}/{new_analysis['total_cells']}"
        )
        print(f"Remaining gaps: {len(new_analysis['gaps'])}")

    if args.generate_only:
        print("\n--generate-only specified, skipping training")
        return

    # Step 5: Training
    # Use combined data only if we generated it this run (not --skip-generation)
    if args.skip_generation:
        training_data = args.data
    else:
        training_data = combined_path if combined_path.exists() else args.data
    run_training(
        training_data,
        args.output,
        args.steps,
        args.batch_size,
        num_generations=args.num_generations,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler=args.lr_scheduler,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        use_batched_eval=not args.no_batch_eval,
        use_cache=not args.no_cache,
        cache_ttl=args.cache_ttl,
        use_async=args.use_async,
        use_pipelined_v2=args.use_pipelined_v2,
    )


if __name__ == "__main__":
    main()
