#!/usr/bin/env python3
"""
Pipelined GRPO Trainer - Uses TRL for C++-optimized model loading, custom loop for async Gemini.
"""

from __future__ import annotations

import os
import gc
import sys
import time
import json
import torch
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from tqdm import tqdm

from datasets import Dataset
from peft import LoraConfig
from transformers import get_scheduler

from trl import GRPOTrainer, GRPOConfig

from src.evaluation.async_reward import PipelinedHybridReward
from src.evaluation.llm_evaluator import LlmEvaluator
from src.evaluation.logging_config import (
    setup_logging,
    get_session_timestamp,

    get_eval_logger,
)


def log(msg):
    """Print with flush for immediate visibility."""
    print(msg, flush=True)


@dataclass
class PipelinedTrainerConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    num_steps: int = 100
    batch_size: int = 1
    num_generations: int = 5
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    max_completion_length: int = 768
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    use_gemini_cache: bool = True
    gemini_cache_ttl: int = 172800
    gemini_model: str = "gemini-3-flash-preview"
    save_steps: int = 50
    output_dir: str = "models/pipelined_output"
    log_steps: int = 10


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


def dummy_reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    return [0.0] * len(completions)


class PipelinedTrainer:
    def __init__(self, config: PipelinedTrainerConfig, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval_logger = get_eval_logger()
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.pipelined_evaluator = None
        self.llm_evaluator = None
        self._trl_trainer = None
        self._pending_step: Optional[StepData] = None
        self.metrics = {"step": [], "loss": [], "reward_mean": [], "reward_std": [], "gen_time": [], "wait_time": [], "update_time": []}
        
    def setup(self) -> None:
        log("=" * 70)
        log("PIPELINED TRAINER SETUP")
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
        self.pipelined_evaluator = PipelinedHybridReward(llm_evaluator=self.llm_evaluator, correctness_weight=0.3, semantic_weight=0.7)
        
        log("Starting async evaluator...")
        self.pipelined_evaluator.start()
        log("  Async evaluator started")
        
        # Now load model (this takes ~33s)
        log("Creating GRPOConfig...")
        model_init_kwargs = {
            "torch_dtype": torch.bfloat16,
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.bfloat16,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "device_map": {"": 0},
            "use_cache": False,
        }
        
        grpo_config = GRPOConfig(
            output_dir=self.config.output_dir,
            max_steps=1,
            per_device_train_batch_size=self.config.batch_size,
            num_generations=self.config.num_generations,
            steps_per_generation=self.config.num_generations,
            max_completion_length=self.config.max_completion_length,
            temperature=self.config.temperature,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            gradient_checkpointing=True,
            remove_unused_columns=False,
            model_init_kwargs=model_init_kwargs,
            report_to="none",
        )
        log("  GRPOConfig OK")
        
        peft_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        log("  LoRA config OK")
        
        log(f"Loading model via TRL: {self.config.model_name}")
        load_start = time.time()
        self._trl_trainer = GRPOTrainer(
            model=self.config.model_name,
            reward_funcs=dummy_reward_fn,
            args=grpo_config,
            train_dataset=self.train_dataset,
            peft_config=peft_config,
        )
        log(f"  TRL loading complete in {time.time() - load_start:.1f}s")
        
        log("Extracting components...")
        self.model = self._trl_trainer.model
        self.tokenizer = self._trl_trainer.processing_class
        
        log("Creating optimizer...")
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=0.01)
        
        num_warmup = int(self.config.num_steps * self.config.warmup_ratio)
        log(f"Creating scheduler (warmup={num_warmup})...")
        self.scheduler = get_scheduler("cosine", optimizer=self.optimizer, num_warmup_steps=num_warmup, num_training_steps=self.config.num_steps)
        
        if torch.cuda.is_available():
            log(f"  VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        log("=" * 70)
        log("Setup complete")
        log("=" * 70)
        
    def cleanup(self) -> None:
        log("Cleaning up...")
        if self.pipelined_evaluator:
            self.pipelined_evaluator.stop()
        if self.llm_evaluator and self.llm_evaluator.cache_active:
            self.llm_evaluator.delete_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
    def train(self) -> Dict[str, List[float]]:
        self.setup()
        try:
            self._train_loop()
        finally:
            self.cleanup()
        return self.metrics
        
    def _train_loop(self) -> None:
        log("")
        log("=" * 70)
        log("TRAINING LOOP")
        log("=" * 70)
        log(f"Steps: {self.config.num_steps}, Batch: {self.config.batch_size}, Gens: {self.config.num_generations}")
        log("")
        
        from torch.utils.data import DataLoader
        dataloader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=True)
        dataloader_iter = iter(dataloader)
        
        pbar = tqdm(range(self.config.num_steps), desc="Training")
        
        for step in pbar:
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)
                
            loss = self._pipeline_step(step, batch)
            if loss is not None:
                pbar.set_postfix({"loss": f"{loss:.4f}"})
            if step > 0 and step % self.config.save_steps == 0:
                self._save_checkpoint(step)
                
        self._drain_pipeline()
        self._save_checkpoint(self.config.num_steps, final=True)
        log("Training complete")
        
    def _pipeline_step(self, step: int, batch: Dict[str, Any]) -> Optional[float]:
        t0 = time.time()
        
        log(f"Step {step}: Generating...")
        gen_start = time.time()
        step_data = self._generate_completions(step, batch)
        gen_time = time.time() - gen_start
        log(f"  Generated {len(step_data.completions)} completions in {gen_time:.1f}s")
        
        log(f"Step {step}: Firing Gemini async...")
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
        log(f"  Fired in {fire_time:.2f}s")
        
        loss = None
        wait_time = 0.0
        update_time = 0.0
        
        if step == 0:
            log(f"Step {step}: Waiting for Gemini (sync, first step)...")
            wait_start = time.time()
            rewards = self.pipelined_evaluator.get_rewards(step)
            wait_time = time.time() - wait_start
            log(f"  Got rewards in {wait_time:.1f}s: {rewards}")
            
            log(f"Step {step}: Updating weights...")
            update_start = time.time()
            loss = self._update_weights(step_data, rewards)
            update_time = time.time() - update_start
            log(f"  Loss: {loss:.4f} in {update_time:.1f}s")
            
        elif self._pending_step is None:
            log(f"Step {step}: Pipeline setup (pending={step})")
            self._pending_step = step_data
            
        else:
            prev = self._pending_step.step_id
            log(f"Step {step}: Waiting for step {prev}'s Gemini...")
            wait_start = time.time()
            prev_rewards = self.pipelined_evaluator.get_rewards(prev)
            wait_time = time.time() - wait_start
            log(f"  Got rewards for step {prev} in {wait_time:.1f}s: {prev_rewards}")
            
            log(f"Step {step}: Updating weights for step {prev}...")
            update_start = time.time()
            loss = self._update_weights(self._pending_step, prev_rewards)
            update_time = time.time() - update_start
            log(f"  Loss: {loss:.4f} in {update_time:.1f}s")
            
            self._pending_step = step_data
            
        total_time = time.time() - t0
        timing = f"gen={gen_time:.1f}s fire={fire_time:.2f}s wait={wait_time:.1f}s update={update_time:.1f}s total={total_time:.1f}s"
        log(f"Step {step} TIMING: {timing}")
        
        self.metrics["step"].append(step)
        self.metrics["gen_time"].append(gen_time)
        self.metrics["wait_time"].append(wait_time)
        self.metrics["update_time"].append(update_time)
        if loss is not None:
            self.metrics["loss"].append(loss)
        return loss
        
    def _drain_pipeline(self) -> None:
        if self._pending_step is None:
            return
        log(f"Draining pipeline: step {self._pending_step.step_id}")
        rewards = self.pipelined_evaluator.get_rewards(self._pending_step.step_id)
        loss = self._update_weights(self._pending_step, rewards)
        log(f"  Drained: loss={loss:.4f}")
        self._pending_step = None
        
    def _generate_completions(self, step: int, batch: Dict[str, Any]) -> StepData:
        prompt = batch["prompt"][0] if isinstance(batch["prompt"], list) else batch["prompt"]
        expected_judgment = self._get_field(batch, "expected_judgment", "Yes")
        domain = self._get_field(batch, "domain", "empirical")
        proposition = self._get_field(batch, "proposition", "")
        evidence = self._get_field(batch, "evidence", "")
        oracle_conditions = self._get_field(batch, "oracle_conditions", "")
        oracle_temporal_context = self._get_field(batch, "oracle_temporal_context", "")
        mode = self._get_field(batch, "mode", "")
        
        # Debug logging removed - trainer retired
        prompt_encoding = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=4096)
        prompt_ids = prompt_encoding["input_ids"].to(self.device)
        prompt_mask = prompt_encoding["attention_mask"].to(self.device)

        
        self.model.gradient_checkpointing_disable()
        self.model.config.use_cache = True

        
        completions = []
        all_completion_ids = []
        
        with torch.no_grad():
            for i in range(self.config.num_generations):
                gen_start = time.time()

                outputs = self.model.generate(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    max_new_tokens=self.config.max_completion_length,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                completion_ids = outputs[0, prompt_ids.size(1):]
                all_completion_ids.append(completion_ids)
                completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
                completions.append(completion_text)
                self.logger.debug(f"[step={step}] _generate_completions: gen {i+1} done in {time.time()-gen_start:.1f}s, {len(completion_text)} chars")
                
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable()
        
        max_len = max(c.size(0) for c in all_completion_ids)
        padded = torch.full((len(all_completion_ids), max_len), self.tokenizer.pad_token_id, device=self.device)
        mask = torch.zeros_like(padded)
        for i, c in enumerate(all_completion_ids):
            padded[i, :c.size(0)] = c
            mask[i, :c.size(0)] = 1
            
        prompt_ids_exp = prompt_ids.expand(self.config.num_generations, -1)
        prompt_mask_exp = prompt_mask.expand(self.config.num_generations, -1)
        
        return StepData(
            step_id=step, prompt=prompt, prompt_ids=prompt_ids_exp, prompt_mask=prompt_mask_exp,
            completion_ids=padded, completion_mask=mask, completions=completions,
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
        self.model.train()
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        mean_reward = rewards_tensor.mean()
        std_reward = rewards_tensor.std() + 1e-8
        advantages = (rewards_tensor - mean_reward) / std_reward
        
        self.metrics["reward_mean"].append(mean_reward.item())
        self.metrics["reward_std"].append(std_reward.item())
        
        input_ids = torch.cat([step_data.prompt_ids, step_data.completion_ids], dim=1)
        attention_mask = torch.cat([step_data.prompt_mask, step_data.completion_mask], dim=1)
        prompt_len = step_data.prompt_ids.size(1)
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        
        completion_start = prompt_len - 1
        mask = (shift_labels != -100) & (shift_labels != self.tokenizer.pad_token_id)
        mask[:, :completion_start] = False
        
        seq_log_probs = (token_log_probs * mask.float()).sum(dim=-1)
        loss = -(advantages * seq_log_probs).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
        
    def _save_checkpoint(self, step: int, final: bool = False) -> None:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = output_dir if final else output_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(checkpoint_dir))
        self.tokenizer.save_pretrained(str(checkpoint_dir))
        state = {"step": step, "optimizer_state_dict": self.optimizer.state_dict(), "scheduler_state_dict": self.scheduler.state_dict(), "metrics": self.metrics}
        torch.save(state, checkpoint_dir / "training_state.pt")
        log(f"*** Checkpoint saved: {checkpoint_dir} ***")


def train_pipelined(
    data_path: str,
    output_dir: str,
    num_steps: int = 100,
    batch_size: int = 1,
    num_generations: int = 5,
    learning_rate: float = 5e-5,
    warmup_ratio: float = 0.1,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    load_in_4bit: bool = True,
    lora_r: int = 64,
    lora_alpha: int = 128,
    use_gemini_cache: bool = True,
    gemini_cache_ttl: int = 172800,
    save_steps: int = 50,
    log_steps: int = 10,
    log_level: str = "info",
    resume_from_checkpoint: Optional[str] = None,
    **kwargs,
) -> Dict[str, List[float]]:
    setup_logging(level=log_level, console_level="info")
    session_ts = get_session_timestamp()
    
    log("=" * 70)
    log("PIPELINED GRPO TRAINER")
    log("=" * 70)
    log(f"Session: {session_ts}")
    log(f"Log files: logs/training_{session_ts}.log, logs/gemini_{session_ts}.log")
    log(f"Data: {data_path}")
    log(f"Output: {output_dir}")
    log(f"Steps: {num_steps}, Batch: {batch_size}, Gens: {num_generations}")
    log(f"LR: {learning_rate}, Model: {model_name}")
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
        model_name=model_name, lora_r=lora_r, lora_alpha=lora_alpha,
        num_steps=num_steps, batch_size=batch_size, num_generations=num_generations,
        learning_rate=learning_rate, warmup_ratio=warmup_ratio,
        use_gemini_cache=use_gemini_cache, gemini_cache_ttl=gemini_cache_ttl,
        save_steps=save_steps, log_steps=log_steps, output_dir=output_dir,
    )
    
    trainer = PipelinedTrainer(config, dataset)
    return trainer.train()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pipelined_trainer.py <data_path>")
        sys.exit(1)
    train_pipelined(sys.argv[1], "models/test", num_steps=3, num_generations=2)
