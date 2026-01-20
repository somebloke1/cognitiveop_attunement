"""
Model management for the Cognitive Operation Attunement system.

Handles:
- Base model loading with quantization
- LoRA adapter loading and hot-swapping
- Inference with active adapter
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training

from .config import config, ProjectConfig
from .cognitive_levels import CognitiveLevel, get_level, LEVELS


class CognitiveModelManager:
    """
    Manages the base model and cognitive operation adapters.

    Architecture:
    - One base model loaded in 4-bit quantization
    - Four LoRA adapters for the four cognitive levels
    - Hot-swapping between adapters at runtime
    """

    def __init__(self, cfg: Optional[ProjectConfig] = None):
        self.cfg = cfg or config
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.loaded_adapters: Dict[str, bool] = {}
        self.active_adapter: Optional[str] = None

    def load_base_model(self) -> None:
        """Load the base model with 4-bit quantization."""

        print(f"Loading base model: {self.cfg.cognitive.base_model}")

        # Configure quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.cfg.hardware.load_in_4bit,
            bnb_4bit_compute_dtype=getattr(torch, self.cfg.hardware.bnb_4bit_compute_dtype),
            bnb_4bit_quant_type=self.cfg.hardware.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.cfg.hardware.bnb_4bit_use_double_quant,
        )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.cognitive.base_model,
            quantization_config=bnb_config,
            device_map=self.cfg.hardware.device,
            torch_dtype=getattr(torch, self.cfg.hardware.dtype),
            trust_remote_code=True,
            token=self.cfg.hf_token,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.cognitive.base_model,
            trust_remote_code=True,
            token=self.cfg.hf_token,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Base model loaded successfully on {self.cfg.hardware.device}")

    def prepare_for_training(self) -> None:
        """Prepare the model for k-bit training."""

        if self.model is None:
            raise RuntimeError("Base model not loaded. Call load_base_model() first.")

        self.model = prepare_model_for_kbit_training(self.model)
        print("Model prepared for k-bit training")

    def create_adapter(self, level: CognitiveLevel) -> None:
        """
        Create a new LoRA adapter for a cognitive level.

        This initializes untrained adapter weights that will be trained
        to embody the operational characteristics of the specified level.
        """

        if self.model is None:
            raise RuntimeError("Base model not loaded. Call load_base_model() first.")

        level_spec = get_level(level)
        adapter_name = level_spec.adapter_name

        lora_config = LoraConfig(
            r=self.cfg.lora.r,
            lora_alpha=self.cfg.lora.lora_alpha,
            lora_dropout=self.cfg.lora.lora_dropout,
            target_modules=self.cfg.lora.target_modules,
            bias=self.cfg.lora.bias,
            task_type=self.cfg.lora.task_type,
        )

        # Add adapter to model
        self.model.add_adapter(lora_config, adapter_name=adapter_name)
        self.loaded_adapters[adapter_name] = True

        print(f"Created adapter '{adapter_name}' for {level_spec.name} level")

    def load_adapter(self, level: CognitiveLevel, adapter_path: Optional[Path] = None) -> None:
        """
        Load a trained LoRA adapter for a cognitive level.

        Args:
            level: The cognitive level to load adapter for
            adapter_path: Path to saved adapter weights (default: use config path)
        """

        if self.model is None:
            raise RuntimeError("Base model not loaded. Call load_base_model() first.")

        level_spec = get_level(level)
        adapter_name = level_spec.adapter_name

        if adapter_path is None:
            adapter_path = self.cfg.models_dir / self.cfg.cognitive.adapters[adapter_name]

        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter not found at {adapter_path}")

        self.model.load_adapter(str(adapter_path), adapter_name=adapter_name)
        self.loaded_adapters[adapter_name] = True

        print(f"Loaded adapter '{adapter_name}' from {adapter_path}")

    def set_adapter(self, level: CognitiveLevel) -> None:
        """
        Set the active adapter for inference.

        This is the hot-swap operation - changes which cognitive operation
        the model performs without reloading the base model.
        """

        if self.model is None:
            raise RuntimeError("Base model not loaded.")

        level_spec = get_level(level)
        adapter_name = level_spec.adapter_name

        if adapter_name not in self.loaded_adapters:
            raise RuntimeError(f"Adapter '{adapter_name}' not loaded. Load it first.")

        self.model.set_adapter(adapter_name)
        self.active_adapter = adapter_name

        print(f"Active adapter set to '{adapter_name}' ({level_spec.name} level)")

    def disable_adapters(self) -> None:
        """Disable all adapters, returning to base model behavior."""

        if self.model is None:
            return

        self.model.disable_adapters()
        self.active_adapter = None
        print("All adapters disabled - using base model")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """
        Generate a response using the currently active adapter.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to sample or use greedy decoding
            **kwargs: Additional generation arguments

        Returns:
            Generated text response
        """

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded.")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the prompt from response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()

        return response

    def cognitive_operation(
        self,
        level: CognitiveLevel,
        input_data: str,
        **generation_kwargs
    ) -> str:
        """
        Perform a cognitive operation at the specified level.

        This is the main interface for using the cognitive system:
        1. Sets the appropriate adapter
        2. Constructs the prompt with level-specific system prompt
        3. Generates the response
        4. Returns the output

        Args:
            level: Which cognitive operation to perform
            input_data: The input to process
            **generation_kwargs: Generation parameters

        Returns:
            The output of the cognitive operation
        """

        level_spec = get_level(level)

        # Set the adapter for this level
        self.set_adapter(level)

        # Construct prompt with system message
        prompt = f"""<|system|>
{level_spec.system_prompt}
<|user|>
{input_data}
<|assistant|>
"""

        return self.generate(prompt, **generation_kwargs)

    def save_adapter(self, level: CognitiveLevel, output_path: Optional[Path] = None) -> None:
        """Save a trained adapter to disk."""

        if self.model is None:
            raise RuntimeError("Model not loaded.")

        level_spec = get_level(level)
        adapter_name = level_spec.adapter_name

        if output_path is None:
            output_path = self.cfg.models_dir / self.cfg.cognitive.adapters[adapter_name]

        output_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_path, selected_adapters=[adapter_name])

        print(f"Saved adapter '{adapter_name}' to {output_path}")

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current GPU memory usage."""

        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}

        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }
