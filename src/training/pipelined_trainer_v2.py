#!/usr/bin/env python3
"""
Pipelined GRPO Trainer V2 - Native model loading (no TRL), custom loop for async Gemini.

Key improvements over V1:
- Loads model directly via transformers + peft (saves ~2GB VRAM, ~30s load time)
- True cross-step pipelining: Gemini N runs while generating N+1
- Memory-efficient chunked forward pass for weight updates
"""

from __future__ import annotations

import gc
import time
import torch
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from tqdm import tqdm

from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_scheduler,
)

from src.evaluation.async_reward import PipelinedHybridReward
from src.evaluation.llm_evaluator import LlmEvaluator
from src.logging import RunLogger, RunConfig, StepMetrics, InferenceRecord

# Reduce CUDA memory fragmentation (must be set before any CUDA allocations)
if torch.cuda.is_available():
    try:
        torch.cuda.memory.set_allocator_settings("expandable_segments:True")
    except Exception:
        pass  # Older PyTorch versions may not support this
from src.evaluation.logging_config import (
    setup_logging,
    get_session_timestamp,
    get_eval_logger,
)

import signal
import atexit
import shutil


# Module-level logger reference (set by trainer after RunLogger init)
_run_logger: Optional['RunLogger'] = None

def log(
    msg: str,
    level: str = "info",
    run_id: Optional[str] = None,
    step_id: Optional[int] = None,
    **context
):
    """
    Log message to both console and structured run log.
    
    Before RunLogger initialization: prints to console only.
    After initialization: writes to both console and logs/runs/*/training.log.
    
    Args:
        msg: Log message
        level: Log level (debug, info, warning, error)
        run_id: Run identifier (auto-included if _run_logger set)
        step_id: Step identifier for step-scoped messages
        **context: Additional structured context (key=value)
    """
    # Build structured prefix
    prefix_parts = []
    if run_id:
        prefix_parts.append(f"run={run_id[-8:]}")
    elif _run_logger:
        prefix_parts.append(f"run={_run_logger.run_id[-8:]}")
    if step_id is not None:
        prefix_parts.append(f"step={step_id}")
    for k, v in context.items():
        prefix_parts.append(f"{k}={v}")
    
    prefix = f"[{' '.join(prefix_parts)}] " if prefix_parts else ""
    full_msg = f"{prefix}{msg}"
    
    print(full_msg, flush=True)
    if _run_logger is not None:
        logger = _run_logger.training
        getattr(logger, level)(full_msg)


@dataclass
class PipelinedTrainerConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    lora_r: int = 96
    lora_alpha: int = 192
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    num_steps: int = 100
    batch_size: int = 4
    num_generations: int = 8
    learning_rate: float = 2e-6
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    max_completion_length: int = 1536
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    use_gemini_cache: bool = True
    gemini_cache_ttl: int = 172800
    gemini_model: str = "gemini-3-flash-preview"
    gemini_max_concurrent: int = 8  # Max parallel Gemini API calls per step (dynamic backoff on 429)
    save_steps: int = 50
    output_dir: str = "models/pipelined_output"
    log_steps: int = 10
    gradient_checkpointing: bool = True


@dataclass
class StepData:
    step_id: int
    prompt: str
    prompt_ids: torch.Tensor
    prompt_mask: torch.Tensor
    completion_ids: torch.Tensor
    completion_mask: torch.Tensor
    completions: List[str]
    expected_judgment: str
    domain: str
    proposition: str
    evidence: str
    oracle_conditions: str
    oracle_temporal_context: str
    mode: str
    fired_at: float = 0.0


class PipelinedTrainerV2:
    """
    Custom GRPO trainer with:
    - Native model loading (no TRL overhead)
    - Cross-step pipelining for async Gemini evaluation
    - Memory-efficient generation and weight updates
    - Checkpoint resumption support
    """
    
    def __init__(self, config: PipelinedTrainerConfig, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None, resume_from_checkpoint: Optional[str] = None, experiment_name: str = "training"):
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.resume_from_checkpoint = resume_from_checkpoint
        self.experiment_name = experiment_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval_logger = get_eval_logger()
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.pipelined_evaluator = None
        self.llm_evaluator = None
        self._pending_step: Optional[StepData] = None
        self._start_step: int = 0  # For checkpoint resumption
        self._rng_seed: int = 42  # Base seed for deterministic sampling
        self.run_logger: Optional[RunLogger] = None  # Structured logging
        self.metrics = {
            "step": [], "loss": [], "reward_mean": [], "reward_std": [],
            "gen_time": [], "wait_time": [], "update_time": [],
            "seq_log_prob_mean": [], "seq_log_prob_std": [],
            "completion_tokens_mean": [], "completion_tokens_std": [],
            "advantages": [], "grad_norm": [],
        }
        
    def setup(self) -> None:
        log("=" * 70)
        log("PIPELINED TRAINER V2 SETUP (Native Loading)")
        log("=" * 70)
        
        # Start async evaluator FIRST - it can initialize while model loads
        log("Creating Gemini evaluator...")
        self.llm_evaluator = LlmEvaluator(
            provider="gemini",
            model_name=self.config.gemini_model,
            use_cache=self.config.use_gemini_cache,
            cache_ttl=self.config.gemini_cache_ttl,
        )
        
        if self.config.use_gemini_cache:
            log("Creating Gemini cache...")
            cache_name = self.llm_evaluator.create_cache()
            log(f"  Cache: {cache_name}")
        
        log("Creating PipelinedHybridReward...")
        # Note: run_id set later after train() generates it
        self.pipelined_evaluator = PipelinedHybridReward(
            llm_evaluator=self.llm_evaluator,
            correctness_weight=0.3,
            semantic_weight=0.7,
            max_concurrent=self.config.gemini_max_concurrent,
            run_id="",  # Will be set in train() once run_id is generated
        )
        
        log("Starting async evaluator...")
        self.pipelined_evaluator.start()
        log("  Async evaluator started")
        
        # Load model natively (no TRL)
        log(f"Loading model natively: {self.config.model_name}")
        load_start = time.time()
        
        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        # Load base model
        log("  Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            trust_remote_code=True,
        )
        
        # Load tokenizer
        log("  Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Prepare for k-bit training
        log("  Preparing for k-bit training...")
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=self.config.gradient_checkpointing,
        )
        
        # Add LoRA adapters
        log("  Adding LoRA adapters...")
        peft_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(base_model, peft_config)
        
        load_time = time.time() - load_start
        log(f"  Model loaded in {load_time:.1f}s")
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        log(f"  Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
        
        if torch.cuda.is_available():
            log(f"  VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # Create optimizer
        log("Creating optimizer...")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        # Create scheduler
        num_warmup = int(self.config.num_steps * self.config.warmup_ratio)
        log(f"Creating scheduler (warmup={num_warmup})...")
        self.scheduler = get_scheduler(
            "cosine",
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup,
            num_training_steps=self.config.num_steps
        )
        
        log("=" * 70)
        log("Setup complete")
        log("=" * 70)
        
    def _load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load checkpoint state for resumption.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            
        Returns:
            Starting step number (step after the checkpointed step)
        """
        checkpoint_dir = Path(checkpoint_path)
        if not checkpoint_dir.exists():
            raise ValueError(f"Checkpoint directory not found: {checkpoint_path}")
        
        log(f"Loading checkpoint from {checkpoint_path}...")
        
        # Load training state
        state_path = checkpoint_dir / "training_state.pt"
        if not state_path.exists():
            raise ValueError(f"training_state.pt not found in {checkpoint_path}")
        
        state = torch.load(state_path, map_location=self.device, weights_only=False)
        checkpoint_step = state["step"]
        log(f"  Checkpoint step: {checkpoint_step}")
        
        # Load adapter weights using PEFT's load_adapter method
        # The checkpoint directory contains adapter_config.json and adapter_model.safetensors
        # which PEFT can load directly
        log("  Loading adapter weights via PEFT...")
        
        # Delete the default adapter and load from checkpoint
        # PEFT creates a "default" adapter, we need to replace it
        adapter_name = "default"
        self.model.delete_adapter(adapter_name)
        self.model.load_adapter(str(checkpoint_dir), adapter_name=adapter_name)
        self.model.set_adapter(adapter_name)
        
        log(f"  Adapter '{adapter_name}' loaded from checkpoint")
        
        # Load optimizer state
        log("  Loading optimizer state...")
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        
        # Load scheduler state
        log("  Loading scheduler state...")
        self.scheduler.load_state_dict(state["scheduler_state_dict"])
        
        # Load metrics (append to existing)
        if "metrics" in state:
            log("  Loading metrics history...")
            checkpoint_metrics = state["metrics"]
            for key in self.metrics:
                if key in checkpoint_metrics:
                    self.metrics[key] = checkpoint_metrics[key].copy() if isinstance(checkpoint_metrics[key], list) else checkpoint_metrics[key]
            log(f"  Restored {len(self.metrics.get('step', []))} metric entries")
        
        start_step = checkpoint_step + 1
        log(f"  Resuming from step {start_step}")
        return start_step
        
    def cleanup(self) -> None:
        log("Cleaning up...")
        # Restore original signal handlers
        if hasattr(self, '_original_sigterm'):
            signal.signal(signal.SIGTERM, self._original_sigterm)
        if hasattr(self, '_original_sigint'):
            signal.signal(signal.SIGINT, self._original_sigint)
        # Unregister atexit handler (normal cleanup path)
        if hasattr(self, '_atexit_cleanup'):
            try:
                atexit.unregister(self._atexit_cleanup)
            except Exception:
                pass
        if self.pipelined_evaluator:
            self.pipelined_evaluator.stop()
        if self.llm_evaluator and self.llm_evaluator.cache_active:
            self.llm_evaluator.delete_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
    def train(self) -> Dict[str, List[float]]:
        self.setup()
        
        # Initialize structured run logger
        # Generate unique run_id and track continuation chain
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._run_id = f"{timestamp}_{self.experiment_name}"
        self._continuation_id = self.resume_from_checkpoint if self.resume_from_checkpoint else None
        
        log(f"  [run_id={self._run_id}] Starting run")
        if self._continuation_id:
            log(f"  [run_id={self._run_id}] Continuing from: {self._continuation_id}")
        
        run_config = RunConfig(
            data_path="",  # Set from train_pipelined_v2 context
            output_dir=self.config.output_dir,
            num_steps=self.config.num_steps,
            batch_size=self.config.batch_size,
            num_generations=self.config.num_generations,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            save_steps=self.config.save_steps,
            model_name=self.config.model_name,
            resume_from_checkpoint=self.resume_from_checkpoint,
            extra={
                "lora_r": self.config.lora_r,
                "lora_alpha": self.config.lora_alpha,
                "gradient_checkpointing": self.config.gradient_checkpointing,
                "max_completion_length": self.config.max_completion_length,
                "gemini_max_concurrent": self.config.gemini_max_concurrent,
            }
        )
        self.run_logger = RunLogger(
            experiment_name=self.experiment_name,
            config=run_config,
            resume_from=self.resume_from_checkpoint,
        )
        
        # Enable structured logging via module-level reference
        global _run_logger
        _run_logger = self.run_logger
        self._run_id = self.run_logger.run_id
        
        self._last_loss = None
        self._last_reward_mean = None
        
        # Register signal handlers for graceful shutdown (SIGTERM from Docker/K8s, SIGINT from Ctrl+C)
        self._original_sigterm = signal.getsignal(signal.SIGTERM)
        self._original_sigint = signal.getsignal(signal.SIGINT)
        
        def graceful_shutdown(signum, frame):
            sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
            step = getattr(self, '_current_step', None)
            log(f"Received {sig_name} - initiating graceful shutdown", step_id=step, level="warning", signal=sig_name)
            # Save emergency checkpoint
            if step is not None and step > self._start_step:
                try:
                    log("Saving emergency checkpoint...", step_id=step)
                    self._save_checkpoint(step, final=False)
                except Exception as e:
                    log(f"Emergency checkpoint failed: {e}", step_id=step, level="error")
            # Re-raise as KeyboardInterrupt to trigger normal exception handling
            raise KeyboardInterrupt(f"Graceful shutdown from {sig_name}")
        
        signal.signal(signal.SIGTERM, graceful_shutdown)
        signal.signal(signal.SIGINT, graceful_shutdown)
        
        # Register atexit as backup cleanup (for unexpected exits)
        def atexit_cleanup():
            log("atexit cleanup triggered")
            try:
                if self.pipelined_evaluator:
                    self.pipelined_evaluator.stop()
                if self.run_logger and not hasattr(self.run_logger, '_finalized'):
                    self.run_logger.finalize(status="crashed", error_message="Unexpected exit (atexit)")
            except Exception as e:
                log(f"atexit cleanup error: {e}")
        
        atexit.register(atexit_cleanup)
        self._atexit_cleanup = atexit_cleanup  # Store reference for unregistration
        
        # Update evaluator with actual run_id and run_logger now that they're generated
        if self.pipelined_evaluator:
            self.pipelined_evaluator.run_id = self._run_id
            self.pipelined_evaluator.set_run_logger(self.run_logger)
        
        try:
            # Load checkpoint after setup (model, optimizer, scheduler must exist)
            if self.resume_from_checkpoint:
                self._start_step = self._load_checkpoint(self.resume_from_checkpoint)
                # Create fresh Gemini cache on resume (old one may have expired)
                if self.config.use_gemini_cache and self.llm_evaluator:
                    log("Creating fresh Gemini cache for resumed training...")
                    if self.llm_evaluator.cache_active:
                        self.llm_evaluator.delete_cache()
                    cache_name = self.llm_evaluator.create_cache()
                    log(f"  New cache: {cache_name}")
            self._train_loop()
            # Finalize with completed status
            if self.run_logger:
                self.run_logger.finalize(
                    status="completed",
                    final_step=self.config.num_steps,
                    final_reward_mean=self._last_reward_mean,
                    final_loss=self._last_loss,
                )
        except KeyboardInterrupt:
            log("\nTraining interrupted by user")
            if self.run_logger:
                self.run_logger.finalize(
                    status="interrupted",
                    final_reward_mean=self._last_reward_mean,
                    final_loss=self._last_loss,
                )
            raise
        except Exception as e:
            log(f"Training failed with error: {e}")
            if self.run_logger:
                self.run_logger.finalize(
                    status="failed",
                    final_reward_mean=self._last_reward_mean,
                    final_loss=self._last_loss,
                    error_message=str(e),
                )
            raise
        finally:
            self.cleanup()
        return self.metrics
        
    def _train_loop(self) -> None:
        log("")
        log("=" * 70)
        log("TRAINING LOOP")
        log("=" * 70)
        
        start_step = self._start_step
        total_steps = self.config.num_steps
        
        if start_step > 0:
            log(f"Resuming from step {start_step}")
        log(f"Steps: {start_step} -> {total_steps}, Batch: {self.config.batch_size}, Gens: {self.config.num_generations}")
        log("")
        
        # Use seeded random sampling for deterministic batch selection
        # This ensures we can skip already-processed batches on resume
        import random
        
        pbar = tqdm(
            range(start_step, total_steps),
            desc="Training",
            initial=start_step,
            total=total_steps
        )
        
        for step in pbar:
            # Deterministic batch selection: seed RNG with base_seed + step
            # This ensures the same batches are selected regardless of resume point
            step_rng = random.Random(self._rng_seed + step)
            batch_indices = step_rng.sample(
                range(len(self.train_dataset)),
                k=min(self.config.batch_size, len(self.train_dataset))
            )
            
            # Build batch from selected indices
            batch = {}
            for key in self.train_dataset.column_names:
                batch[key] = [self.train_dataset[i][key] for i in batch_indices]
                
            loss = self._pipeline_step(step, batch)
            if loss is not None:
                pbar.set_postfix({"loss": f"{loss:.4f}"})
            if step > 0 and step % self.config.save_steps == 0:
                self._save_checkpoint(step)
                
        self._drain_pipeline()
        self._save_checkpoint(total_steps, final=True)
        log("Training complete")
        
    def _pipeline_step(self, step: int, batch: Dict[str, Any]) -> Optional[float]:
        """
        Execute one step with cross-step pipelining:
        - Generate completions for current step
        - Fire async Gemini evaluation
        - If not first step, wait for PREVIOUS step's Gemini (should be ready)
        - Update weights for previous step
        """
        t0 = time.time()
        
        log("Generating...", step_id=step)
        gen_start = time.time()
        step_data = self._generate_completions(step, batch)
        gen_time = time.time() - gen_start
        log(f"Generated {len(step_data.completions)} completions in {gen_time:.1f}s", step_id=step)
        
        log("Firing Gemini async...", step_id=step)
        fire_start = time.time()
        self.pipelined_evaluator.fire_evaluation(
            step_id=step,
            completions=step_data.completions,
            proposition=step_data.proposition,
            evidence=step_data.evidence,
            oracle_conditions=step_data.oracle_conditions,
            oracle_temporal_context=step_data.oracle_temporal_context,
            expected_judgment=step_data.expected_judgment,
            domain=step_data.domain,
            mode=step_data.mode,
        )
        step_data.fired_at = time.time()
        fire_time = time.time() - fire_start
        log(f"Fired in {fire_time:.2f}s", step_id=step)
        
        loss = None
        wait_time = 0.0
        update_time = 0.0
        
        if step == 0:
            # First step: must wait synchronously (nothing to overlap with)
            log("Waiting for Gemini (sync, first step)...", step_id=step)
            wait_start = time.time()
            rewards = self.pipelined_evaluator.get_rewards(step)
            wait_time = time.time() - wait_start
            
            if rewards is None:
                # Gemini parse failed - skip this step
                log("SKIPPED (Gemini parse failure)", step_id=step, level="warning")
                del step_data
            else:
                log(f"Got rewards in {wait_time:.1f}s: {rewards}", step_id=step)
                
                log("Updating weights...", step_id=step)
                update_start = time.time()
                loss = self._update_weights(step_data, rewards)
                update_time = time.time() - update_start
                log(f"Loss: {loss:.4f} in {update_time:.1f}s", step_id=step)
                
                # Log structured metrics for step 0
                step_total = time.time() - t0
                self._log_step_completed(
                    step=step,
                    rewards=rewards,
                    loss=loss,
                    gen_time=gen_time,
                    wait_time=wait_time,
                    update_time=update_time,
                    total_time=step_total,
                )
                
                # Step 0 data no longer needed
                del step_data
            
        elif self._pending_step is None:
            # Second step: setup pipeline (no previous to update yet)
            log(f"Pipeline setup (pending={step})", step_id=step)
            self._pending_step = step_data
            
        else:
            # Third+ step: true pipelining
            prev = self._pending_step.step_id
            log(f"Waiting for step {prev}'s Gemini...", step_id=step)
            wait_start = time.time()
            prev_rewards = self.pipelined_evaluator.get_rewards(prev)
            wait_time = time.time() - wait_start
            
            if prev_rewards is None:
                # Gemini parse failed for previous step - skip its update
                log("SKIPPED (Gemini parse failure)", step_id=prev, level="warning")
            else:
                log(f"Got rewards in {wait_time:.1f}s: {prev_rewards}", step_id=prev)
                
                log(f"Updating weights for step {prev}...", step_id=step)
                update_start = time.time()
                loss = self._update_weights(self._pending_step, prev_rewards)
                update_time = time.time() - update_start
                log(f"Loss: {loss:.4f} in {update_time:.1f}s", step_id=prev)
                
                # Log structured metrics for the PREVIOUS step (whose weights we just updated)
                # Note: gen_time for prev step was captured when prev was current - we don't have it here
                # So we log with wait_time and update_time only (gen_time=0 as placeholder)
                self._log_step_completed(
                    step=prev,
                    rewards=prev_rewards,
                    loss=loss,
                    gen_time=0.0,  # Not available for pipelined steps
                    wait_time=wait_time,
                    update_time=update_time,
                    total_time=wait_time + update_time,
                )
            
            # CRITICAL: Explicitly delete old pending step's tensors before reassignment
            # This prevents memory accumulation from unreferenced StepData objects
            old_pending = self._pending_step
            self._pending_step = step_data
            del old_pending  # Release tensors from previous step
            
        total_time = time.time() - t0
        timing = f"gen={gen_time:.1f}s fire={fire_time:.2f}s wait={wait_time:.1f}s update={update_time:.1f}s total={total_time:.1f}s"
        log(f"TIMING: {timing}", step_id=step)
        
        self.metrics["step"].append(step)
        self.metrics["gen_time"].append(gen_time)
        self.metrics["wait_time"].append(wait_time)
        self.metrics["update_time"].append(update_time)
        # Track which step's weights were updated (may differ from current step due to pipelining)
        if loss is not None:
            self.metrics["loss"].append(loss)
        else:
            # No update this step - append NaN for alignment
            self.metrics["loss"].append(float('nan'))
            self.metrics["seq_log_prob_mean"].append(float('nan'))
            self.metrics["seq_log_prob_std"].append(float('nan'))
            self.metrics["completion_tokens_mean"].append(float('nan'))
            self.metrics["completion_tokens_std"].append(float('nan'))
            self.metrics["advantages"].append([])
            self.metrics["grad_norm"].append(float('nan'))
        return loss
        
    def _drain_pipeline(self) -> None:
        """Process any remaining pending step after main loop."""
        if self._pending_step is None:
            return
        drain_step = self._pending_step.step_id
        log("Draining pipeline...", step_id=drain_step)
        
        wait_start = time.time()
        rewards = self.pipelined_evaluator.get_rewards(drain_step)
        wait_time = time.time() - wait_start
        
        if rewards is None:
            log("SKIPPED (Gemini parse failure)", step_id=drain_step, level="warning")
        else:
            update_start = time.time()
            loss = self._update_weights(self._pending_step, rewards)
            update_time = time.time() - update_start
            log(f"Drained: loss={loss:.4f}", step_id=drain_step)
            
            # Log structured metrics for drained step
            self._log_step_completed(
                step=drain_step,
                rewards=rewards,
                loss=loss,
                gen_time=0.0,  # Not tracked for drain
                wait_time=wait_time,
                update_time=update_time,
                total_time=wait_time + update_time,
            )
        self._pending_step = None
    
    def _log_step_completed(
        self,
        step: int,
        rewards: List[float],
        loss: float,
        gen_time: float,
        wait_time: float,
        update_time: float,
        total_time: float,
    ) -> None:
        """
        Log structured metrics for a completed step.
        
        Called after _update_weights() succeeds with all timing/reward data.
        """
        if not self.run_logger:
            return
        
        # Get last-appended diagnostic metrics from self.metrics
        seq_log_prob_mean = self.metrics["seq_log_prob_mean"][-1] if self.metrics["seq_log_prob_mean"] else None
        seq_log_prob_std = self.metrics["seq_log_prob_std"][-1] if self.metrics["seq_log_prob_std"] else None
        token_count_mean = self.metrics["completion_tokens_mean"][-1] if self.metrics["completion_tokens_mean"] else None
        token_count_std = self.metrics["completion_tokens_std"][-1] if self.metrics["completion_tokens_std"] else None
        advantages = self.metrics["advantages"][-1] if self.metrics["advantages"] else None
        grad_norm = self.metrics["grad_norm"][-1] if self.metrics["grad_norm"] else None
        
        # Count correct judgments (from rewards - high reward ~ correct)
        correct_count = sum(1 for r in rewards if r > 0.5)  # Rough heuristic
        
        # Track for finalization
        self._last_loss = loss
        self._last_reward_mean = sum(rewards) / len(rewards) if rewards else None
        
        metrics = StepMetrics(
            step=step,
            reward_mean=self._last_reward_mean,
            reward_std=(sum((r - self._last_reward_mean)**2 for r in rewards) / len(rewards))**0.5 if rewards and len(rewards) > 1 else 0.0,
            reward_min=min(rewards) if rewards else None,
            reward_max=max(rewards) if rewards else None,
            rewards=rewards,
            correct_count=correct_count,
            total_count=len(rewards),
            loss=loss,
            grad_norm=grad_norm if isinstance(grad_norm, (int, float)) and grad_norm == grad_norm else None,  # NaN check
            learning_rate=self.scheduler.get_last_lr()[0] if self.scheduler else None,
            gen_time_s=gen_time,
            eval_time_s=wait_time,
            update_time_s=update_time,
            total_time_s=total_time,
            seq_log_prob_mean=seq_log_prob_mean if seq_log_prob_mean == seq_log_prob_mean else None,  # NaN check
            seq_log_prob_std=seq_log_prob_std if seq_log_prob_std == seq_log_prob_std else None,
            token_count_mean=token_count_mean if token_count_mean == token_count_mean else None,
            token_count_std=token_count_std if token_count_std == token_count_std else None,
            advantages=advantages if advantages else None,
            eval_method="parallel",
            eval_success=True,
        )
        
        self.run_logger.log_step_metrics(metrics)
        
    def _generate_completions(self, step: int, batch: Dict[str, Any]) -> StepData:
        """Generate completions using direct model.generate (memory efficient)."""
        prompt = batch["prompt"][0] if isinstance(batch["prompt"], list) else batch["prompt"]
        expected_judgment = self._get_field(batch, "expected_judgment", "Yes")
        domain = self._get_field(batch, "domain", "empirical")
        proposition = self._get_field(batch, "proposition", "")
        evidence = self._get_field(batch, "evidence", "")
        oracle_conditions = self._get_field(batch, "oracle_conditions", "")
        oracle_temporal_context = self._get_field(batch, "oracle_temporal_context", "")
        mode = self._get_field(batch, "mode", "")
        
        log(f"  [gen] step={step} Generating {self.config.num_generations} local completions...")
        
        # Disable gradient checkpointing for generation (incompatible with use_cache=True)
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_disable()
            log(f"  [gen] Disabled gradient checkpointing: is_gradient_checkpointing={self.model.is_gradient_checkpointing}")
        
        # Tokenize prompt
        prompt_encoding = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        prompt_ids_single = prompt_encoding["input_ids"].to(self.device)
        prompt_mask_single = prompt_encoding["attention_mask"].to(self.device)
        
        # Batch the prompts for efficient generation
        prompt_ids_batched = prompt_ids_single.expand(self.config.num_generations, -1)
        prompt_mask_batched = prompt_mask_single.expand(self.config.num_generations, -1)
        
        gen_start = time.time()
        
        # Generate with model in eval mode, no gradients
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=prompt_ids_batched,
                attention_mask=prompt_mask_batched,
                max_new_tokens=self.config.max_completion_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,  # Critical for speed
            )
        
        gen_time = time.time() - gen_start
        
        # Extract completions (everything after prompt)
        prompt_len = prompt_ids_single.size(1)
        completion_ids = outputs[:, prompt_len:]
        
        # Decode completions
        completions = [self.tokenizer.decode(c, skip_special_tokens=True) for c in completion_ids]
        
        # Log each completion with inference_id: run.step.L.idx (L=Local)
        run_suffix = getattr(self, '_run_id', 'norun')[-6:]
        run_id_full = getattr(self, '_run_id', 'norun')
        local_inference_ids = []
        
        for i, (comp, comp_ids) in enumerate(zip(completions, completion_ids)):
            comp_tokens = (comp_ids != self.tokenizer.pad_token_id).sum().item()
            inference_id = f"{run_suffix}.{step}.L.{i}"
            local_inference_ids.append(inference_id)
            log(f"  [inf_id={inference_id}] LOCAL: {comp_tokens} tokens, {len(comp)} chars")
            
            # Log structured inference record
            if self.run_logger:
                from datetime import datetime
                end_time = datetime.now()
                # Approximate start time based on even distribution
                start_offset = (gen_time / len(completions)) * i
                start_time = datetime.fromtimestamp(gen_start + start_offset)
                
                # Save full prompt + completion content
                content_path = self.run_logger.save_content(
                    inference_id=inference_id,
                    content_type="full",
                    content=f"=== PROMPT ===\n{prompt}\n\n=== COMPLETION ===\n{comp}",
                )
                
                record = InferenceRecord(
                    run_id=run_id_full,
                    step_id=step,
                    inference_id=inference_id,
                    inference_type="local",
                    inference_idx=i,
                    start_time=start_time.isoformat(),
                    end_time=end_time.isoformat(),
                    duration_s=gen_time / len(completions),  # Approximate per-completion
                    status="completed",
                    token_count=comp_tokens,
                    char_count=len(comp),
                    content_path=content_path,
                )
                self.run_logger.log_inference(record)
        
        avg_tokens = sum((c != self.tokenizer.pad_token_id).sum().item() for c in completion_ids) / len(completions)
        log(f"  [run={run_suffix}][step={step}] LOCAL_BATCH: n={len(completions)}, avg_tokens={avg_tokens:.0f}, time={gen_time:.1f}s")
        
        # Expand prompt_ids to match batch size (clone to avoid shared memory)
        prompt_ids = prompt_ids_batched.clone()
        prompt_mask = prompt_mask_batched.clone()
        
        # Create completion mask
        completion_mask = (completion_ids != self.tokenizer.pad_token_id).long()
        
        # Clear cache - gc.collect() BEFORE empty_cache() to release Python refs first
        del outputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Re-enable gradient checkpointing for training
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            
        return StepData(
            step_id=step, prompt=prompt, prompt_ids=prompt_ids, prompt_mask=prompt_mask,
            completion_ids=completion_ids, completion_mask=completion_mask, completions=completions,
            expected_judgment=expected_judgment, domain=domain, proposition=proposition,
            evidence=evidence, oracle_conditions=oracle_conditions,
            oracle_temporal_context=oracle_temporal_context, mode=mode,
        )
        
    def _get_field(self, batch: Dict[str, Any], key: str, default: str = "") -> str:
        value = batch.get(key, default)
        if isinstance(value, list):
            return value[0] if value else default
        return value if value else default
        
    def _update_weights(self, step_data: StepData, rewards: List[float]) -> float:
        """Update weights using GRPO-style relative advantages."""
        self.model.train()
        
        # Ensure gradient checkpointing is enabled for memory-efficient backward pass
        if self.config.gradient_checkpointing and not self.model.is_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # Compute advantages across all completions (GRPO core idea)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        mean_reward = rewards_tensor.mean()
        std_reward = rewards_tensor.std() + 1e-8
        advantages = (rewards_tensor - mean_reward) / std_reward
        
        self.metrics["reward_mean"].append(mean_reward.item())
        self.metrics["reward_std"].append(std_reward.item())
        
        # Build full input sequences
        input_ids = torch.cat([step_data.prompt_ids, step_data.completion_ids], dim=1)
        attention_mask = torch.cat([step_data.prompt_mask, step_data.completion_mask], dim=1)
        prompt_len = step_data.prompt_ids.size(1)
        
        # Create labels (mask prompt tokens)
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100
        
        # Forward pass with memory-efficient chunking for larger batches
        num_completions = input_ids.size(0)
        
        if num_completions <= 2:
            # Small batch - process all at once
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        else:
            # Larger batch - accumulate logits to save memory
            logits_list = []
            chunk_size = 2
            for i in range(0, num_completions, chunk_size):
                chunk_end = min(i + chunk_size, num_completions)
                chunk_output = self.model(
                    input_ids=input_ids[i:chunk_end],
                    attention_mask=attention_mask[i:chunk_end]
                )
                logits_list.append(chunk_output.logits)
                del chunk_output
            logits = torch.cat(logits_list, dim=0)
            del logits_list
        
        # Compute per-token log probs for completion tokens only
        shift_logits = logits[:, prompt_len-1:-1, :].contiguous()
        shift_labels = labels[:, prompt_len:].contiguous()
        
        # Free logits memory
        del logits
        
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        del shift_logits
        
        token_log_probs = torch.gather(
            log_probs, dim=-1,
            index=shift_labels.clamp(min=0).unsqueeze(-1)
        ).squeeze(-1)
        del log_probs
        
        # Mask padding
        mask = (shift_labels != -100) & (shift_labels != self.tokenizer.pad_token_id)
        
        # Sum log probs per sequence
        seq_log_probs = (token_log_probs * mask.float()).sum(dim=-1) / (mask.sum(dim=-1).float().clamp(min=1) ** 0.8)
        
        # Diagnostic: token counts per completion
        token_counts = mask.sum(dim=-1).float()
        
        # GRPO loss: weighted by advantages
        loss = -(advantages * seq_log_probs).mean()
        
        # Backward and update
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        
        # Store diagnostic metrics
        self.metrics["seq_log_prob_mean"].append(seq_log_probs.mean().item())
        self.metrics["seq_log_prob_std"].append(seq_log_probs.std().item())
        self.metrics["completion_tokens_mean"].append(token_counts.mean().item())
        self.metrics["completion_tokens_std"].append(token_counts.std().item())
        self.metrics["advantages"].append(advantages.tolist())
        self.metrics["grad_norm"].append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
        
        # Log diagnostics
        log(f"  [diag] seq_log_probs: mean={seq_log_probs.mean().item():.2f}, std={seq_log_probs.std().item():.2f}")
        log(f"  [diag] token_counts: mean={token_counts.mean().item():.0f}, std={token_counts.std().item():.1f}")
        log(f"  [diag] advantages: {[f'{a:.2f}' for a in advantages.tolist()]}")
        log(f"  [diag] grad_norm: {grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm:.4f}")
        
        # Cleanup - delete ALL intermediate tensors, gc.collect() BEFORE empty_cache()
        loss_value = loss.item()
        del loss, token_log_probs, seq_log_probs, mask, token_counts
        del input_ids, attention_mask, labels, shift_labels
        del rewards_tensor, advantages, mean_reward, std_reward
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return loss_value
        
    def _save_checkpoint(self, step: int, final: bool = False) -> None:
        """
        Save checkpoint with atomic writes to prevent corruption on crash.
        
        Uses tmp directory + rename pattern for atomicity.
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_name = "final" if final else f"checkpoint-{step}"
        checkpoint_dir = output_dir if final else output_dir / checkpoint_name
        tmp_checkpoint_dir = output_dir / f".tmp_{checkpoint_name}"
        
        # Clean up any leftover tmp dir from previous crash
        if tmp_checkpoint_dir.exists():
            shutil.rmtree(tmp_checkpoint_dir)
        
        # Save to temporary directory first
        tmp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(tmp_checkpoint_dir))
        self.tokenizer.save_pretrained(str(tmp_checkpoint_dir))
        
        state = {
            "step": step,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": self.metrics
        }
        torch.save(state, tmp_checkpoint_dir / "training_state.pt")
        
        # Atomic rename (or replace for final checkpoint)
        if final:
            # For final, we save directly to output_dir, so just move contents
            for item in tmp_checkpoint_dir.iterdir():
                dest = checkpoint_dir / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(dest))
            tmp_checkpoint_dir.rmdir()
        else:
            # For intermediate checkpoints, atomic rename of entire directory
            if checkpoint_dir.exists():
                # Remove old checkpoint first
                shutil.rmtree(checkpoint_dir)
            shutil.move(str(tmp_checkpoint_dir), str(checkpoint_dir))
        
        log(f"Checkpoint saved: {checkpoint_dir}", step_id=step, checkpoint=str(checkpoint_dir))
        
        # Track current step for emergency saves
        self._current_step = step
        
        # Log to structured run logger
        if self.run_logger:
            self.run_logger.log_checkpoint(str(checkpoint_dir), step)


def train_pipelined_v2(
    data_path: str,
    output_dir: str,
    num_steps: int = 100,
    batch_size: int = 4,
    num_generations: int = 8,
    learning_rate: float = 2e-6,
    warmup_ratio: float = 0.1,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    load_in_4bit: bool = True,
    lora_r: int = 96,
    lora_alpha: int = 192,
    use_gemini_cache: bool = True,
    gemini_cache_ttl: int = 172800,
    gemini_max_concurrent: int = 8,
    save_steps: int = 50,
    log_steps: int = 10,
    log_level: str = "info",
    gradient_checkpointing: bool = True,
    resume_from_checkpoint: Optional[str] = None,
    **kwargs,
) -> Dict[str, List[float]]:
    """
    Main entry point for pipelined GRPO training.
    
    Args:
        data_path: Path to JSONL training data
        output_dir: Directory for checkpoints and final model
        num_steps: Total training steps
        batch_size: Examples per step (prompts, not completions)
        num_generations: Completions per prompt for GRPO
        learning_rate: Initial learning rate
        warmup_ratio: Fraction of steps for LR warmup
        model_name: HuggingFace model ID
        load_in_4bit: Use 4-bit quantization (required for 24GB VRAM)
        lora_r: LoRA rank
        lora_alpha: LoRA alpha (typically 2x rank)
        use_gemini_cache: Enable Gemini context caching
        gemini_cache_ttl: Cache TTL in seconds
        gemini_max_concurrent: Max parallel Gemini API calls (default 8, dynamic backoff on 429)
        save_steps: Save checkpoint every N steps
        log_steps: Log metrics every N steps
        log_level: Logging verbosity
        gradient_checkpointing: Enable gradient checkpointing (saves memory)
        resume_from_checkpoint: Path to checkpoint to resume from
        
    Returns:
        Dictionary of training metrics
    """
    setup_logging(level=log_level, console_level=log_level)
    session_ts = get_session_timestamp()
    
    log("=" * 70)
    log("PIPELINED GRPO TRAINER V2 (Native Loading)")
    log("=" * 70)
    log(f"Session: {session_ts}")
    log(f"Log files: logs/training_{session_ts}.log, logs/gemini_{session_ts}.log")
    log(f"Data: {data_path}")
    log(f"Output: {output_dir}")
    log(f"Steps: {num_steps}, Batch: {batch_size}, Gens: {num_generations}")
    log(f"LR: {learning_rate}, Model: {model_name}")
    log(f"Gemini: max_concurrent={gemini_max_concurrent}, cache={use_gemini_cache}")
    log("=" * 70)
    
    log("Loading dataset...")
    raw_dataset = Dataset.from_json(data_path)
    log(f"  Loaded {len(raw_dataset)} examples")
    
    def process_example(ex: Dict[str, Any]) -> Dict[str, Any]:
        msgs = ex.get("messages", [])
        meta = ex.get("metadata", {})
        system = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        
        proposition, evidence, conditions = "", "", ""
        lines = user.split("\n")
        section = None
        cond_lines = []
        for line in lines:
            s = line.strip()
            if s.startswith("Proposition:"): section, proposition = "p", s.replace("Proposition:", "").strip()
            elif s.startswith("Evidence:"): section, evidence = "e", s.replace("Evidence:", "").strip()
            elif s.startswith("Conditions for truth:"): section = "c"
            elif s.startswith("Temporal context:"): section = None
            elif section == "c" and s.startswith("-"): cond_lines.append(s)
            elif section == "e" and s and not s.startswith(("Proposition:", "Evidence:", "Conditions", "Temporal")): evidence += " " + s
        conditions = "\n".join(cond_lines)
        
        jmap = {"yes": "Yes", "no": "No", "insufficient": "Insufficient", "Yes": "Yes", "No": "No", "Insufficient": "Insufficient"}
        return {
            "prompt": prompt,
            "expected_judgment": jmap.get(meta.get("expected_judgment", meta.get("judgment_type", "")), "Yes"),
            "domain": meta.get("domain", "empirical"),
            "proposition": proposition,
            "evidence": evidence,
            "oracle_conditions": meta.get("oracle_conditions", conditions),
            "oracle_temporal_context": meta.get("oracle_temporal_context", ""),
            "conditions": conditions,
            "mode": meta.get("mode", ""),
        }
    
    log("Processing dataset...")
    dataset = raw_dataset.map(process_example, remove_columns=raw_dataset.column_names)
    log(f"  Processed {len(dataset)} examples")
    
    config = PipelinedTrainerConfig(
        model_name=model_name,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        num_steps=num_steps,
        batch_size=batch_size,
        num_generations=num_generations,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        use_gemini_cache=use_gemini_cache,
        gemini_cache_ttl=gemini_cache_ttl,
        gemini_max_concurrent=gemini_max_concurrent,
        save_steps=save_steps,
        log_steps=log_steps,
        output_dir=output_dir,
        gradient_checkpointing=gradient_checkpointing,
    )
    
    trainer = PipelinedTrainerV2(config, dataset, resume_from_checkpoint=resume_from_checkpoint)
    return trainer.train()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pipelined_trainer_v2.py <data_path>")
        sys.exit(1)
    train_pipelined_v2(sys.argv[1], "models/test", num_steps=3, num_generations=2)
