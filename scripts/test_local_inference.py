#!/usr/bin/env python3
"""
Test Local Model Inference

Tests whether we can run inference with the local Qwen model
that will be used for GRPO training.
"""

import os
import sys
from pathlib import Path
import json
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")


def test_transformers_inference():
    """Test direct inference with transformers."""
    print("=" * 70)
    print("TRANSFORMERS INFERENCE TEST")
    print("=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_name = "Qwen/Qwen2.5-7B-Instruct"

    print(f"\nLoading model: {model_name}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Quantization config for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Model loaded. Device: {model.device}")
    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # Test inference
    print("\n" + "-" * 70)
    print("INFERENCE TEST")
    print("-" * 70)

    # Simple judgment test
    system_prompt = """You are a Judgment cognitive operation (krisis/mishpat/viveka).
Evaluate whether a proposition is true by grasping the virtually unconditioned.
If conditions are fulfilled, affirm (Yes). If identifiably not fulfilled, deny (No). If cannot determine, withhold (Insufficient)."""

    user_prompt = """Proposition: The sum of angles in a triangle is 180 degrees.

Evidence: By Euclidean geometry, the sum of interior angles of a polygon with n sides is (n-2) × 180°. A triangle has n=3 sides, so (3-2) × 180° = 180°.

Conditions for truth:
- The geometry is Euclidean
- The shape is a valid triangle
- The arithmetic is correct

What is your judgment?"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    print(f"\nInput prompt length: {len(system_prompt) + len(user_prompt)} chars")

    # Format for Qwen
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print("Generating response...")
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print("\n" + "=" * 70)
    print("MODEL RESPONSE")
    print("=" * 70)
    print(response)

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Check if response contains expected judgment
    response_lower = response.lower()
    if "yes" in response_lower and "fulfilled" in response_lower:
        print("✓ Model correctly affirmed the proposition")
    elif "insufficient" in response_lower:
        print("⚠ Model withheld judgment (unexpected for this case)")
    elif "no" in response_lower:
        print("✗ Model incorrectly denied the proposition")
    else:
        print("? Model response unclear")

    # Clean up
    del model
    torch.cuda.empty_cache()
    print(f"\nMemory after cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")


def test_dspy_inference():
    """Test inference through DSPy."""
    print("\n" + "=" * 70)
    print("DSPY LOCAL INFERENCE TEST")
    print("=" * 70)

    import dspy

    # Try to configure local model through DSPy
    try:
        # Note: This may require vLLM or HuggingFace backend
        lm = dspy.LM(
            "huggingface/Qwen/Qwen2.5-7B-Instruct",
            device="cuda",
        )
        dspy.configure(lm=lm)

        from src.dspy_cognitive.modules import JudgmentModule

        module = JudgmentModule(use_chain_of_thought=True)

        result = module(
            proposition="The sum of angles in a triangle is 180 degrees.",
            evidence="By Euclidean geometry, (n-2) × 180° for n-sided polygon. Triangle has n=3.",
            conditions=["Geometry is Euclidean", "Shape is valid triangle", "Arithmetic correct"],
            domain="mathematical",
        )

        print(f"\nJudgment: {result.judgment}")
        print(f"Reasoning: {result.reasoning[:200]}...")

    except Exception as e:
        print(f"\nDSPy local inference failed: {e}")
        print("This is expected - DSPy may require vLLM backend for local models.")
        print("GRPO training handles this internally.")


def main():
    print("Testing Local Model Inference")
    print("Target: Qwen/Qwen2.5-7B-Instruct with 4-bit quantization")
    print()

    # Test 1: Direct transformers inference
    test_transformers_inference()

    # Test 2: DSPy inference (may fail without vLLM)
    # test_dspy_inference()


if __name__ == "__main__":
    main()
