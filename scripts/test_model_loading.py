#!/usr/bin/env python3
"""
Test script to verify base model loading and VRAM usage.

This confirms that our chosen base model fits within the RTX 3090's 24GB VRAM
when loaded with 4-bit quantization (QLoRA-ready).
"""

import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training


def get_gpu_memory():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "total_gb": round(total, 2),
            "free_gb": round(total - reserved, 2),
        }
    return None


def test_model_loading(model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
    """Test loading the base model with 4-bit quantization."""

    print(f"\n{'='*60}")
    print(f"Testing Model: {model_name}")
    print(f"{'='*60}")

    # Initial memory state
    print("\n[1] Initial GPU Memory:")
    mem = get_gpu_memory()
    print(f"    Allocated: {mem['allocated_gb']} GB")
    print(f"    Reserved:  {mem['reserved_gb']} GB")
    print(f"    Total:     {mem['total_gb']} GB")

    # Configure 4-bit quantization
    print("\n[2] Configuring 4-bit quantization (BitsAndBytes)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer
    print("\n[3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"    Vocab size: {tokenizer.vocab_size}")

    # Load model with quantization
    print("\n[4] Loading model with 4-bit quantization...")
    print("    (This may take a few minutes on first download)")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Memory after loading
    print("\n[5] GPU Memory After Model Load:")
    mem = get_gpu_memory()
    print(f"    Allocated: {mem['allocated_gb']} GB")
    print(f"    Reserved:  {mem['reserved_gb']} GB")
    print(f"    Free:      {mem['free_gb']} GB")

    # Prepare for training (QLoRA)
    print("\n[6] Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)

    # Memory after preparation
    print("\n[7] GPU Memory After Training Preparation:")
    mem = get_gpu_memory()
    print(f"    Allocated: {mem['allocated_gb']} GB")
    print(f"    Reserved:  {mem['reserved_gb']} GB")
    print(f"    Free:      {mem['free_gb']} GB")

    # Test inference
    print("\n[8] Testing inference...")
    test_prompt = "The four levels of cognitive operation are:"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n    Prompt: {test_prompt}")
    print(f"    Response: {response}")

    # Final memory state
    print("\n[9] Final GPU Memory:")
    mem = get_gpu_memory()
    print(f"    Allocated: {mem['allocated_gb']} GB")
    print(f"    Reserved:  {mem['reserved_gb']} GB")
    print(f"    Free:      {mem['free_gb']} GB")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Parameters: {model.num_parameters():,}")
    print(f"VRAM Used: ~{mem['reserved_gb']} GB")
    print(f"VRAM Free: ~{mem['free_gb']} GB")
    print(f"Status: {'PASS - Sufficient headroom for LoRA training' if mem['free_gb'] > 5 else 'WARNING - Limited headroom'}")
    print(f"{'='*60}\n")

    # Cleanup
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return True


if __name__ == "__main__":
    import sys

    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-7B-Instruct"
    test_model_loading(model)
