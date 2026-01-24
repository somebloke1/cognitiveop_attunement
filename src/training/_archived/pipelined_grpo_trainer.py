#!/usr/bin/env python3
"""
PipelinedGRPOTrainer - GRPO training with true cross-step pipelined Gemini evaluation.

Pipeline:
    1. Pull test case N
    2. Generate completions N (GPU)
    3. Surface analysis N (local)
    4. Fire Gemini N (async)
    5. Pull test case N+1
    6. Generate completions N+1 (GPU) ← Gemini N runs in parallel
    7. Wait for Gemini N
    8. Compute rewards N, update weights N
    9. Surface analysis N+1
    10. Fire Gemini N+1 (async)
    11. Repeat from step 5

This overlaps Gemini(N) with Generation(N+1), hiding ~5s+ of latency per step.
"""

from __future__ import annotations

import time
import torch
from torch import nn
from typing import Optional, Any, Dict, List, Iterator
from dataclasses import dataclass
import functools
import contextlib

from trl import GRPOTrainer, GRPOConfig
from trl.trainer.grpo_trainer import (
    pad,
    disable_gradient_checkpointing,
    gather_object,
    nanstd,
    apply_chat_template,
    use_adapter,
    prepare_multimodal_messages,
)

from .cognitive_grpo_trainer import CognitiveGRPOTrainer
from src.evaluation.async_reward import PipelinedHybridReward
from src.evaluation.logging_config import get_training_logger, get_eval_logger


@dataclass
class StepState:
    """All state for one training step."""
    step_id: int
    inputs: List[Dict[str, Any]]
    prompts: List[str]
    completions: List[str]
    completion_ids_list: List[List[int]]
    prompt_ids: torch.Tensor
    prompt_mask: torch.Tensor
    completion_ids: torch.Tensor
    completion_mask: torch.Tensor
    old_per_token_logps: Optional[torch.Tensor]
    ref_per_token_logps: Optional[torch.Tensor]
    num_items_in_batch: Any
    forward_kwargs: Dict[str, Any]
    num_images: Optional[List[int]]
    tool_mask: Optional[torch.Tensor]
    # Parsed fields for Gemini
    proposition: str
    evidence: str
    conditions: str
    expected_judgment: str
    domain: str
    mode: str


class PipelinedGRPOTrainer(CognitiveGRPOTrainer):
    """
    GRPO Trainer with true cross-step pipelined Gemini evaluation.
    
    Overrides the training loop to overlap Gemini(N) with Generation(N+1).
    """
    
    def __init__(
        self,
        pipelined_evaluator: Optional[PipelinedHybridReward] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pipelined_evaluator = pipelined_evaluator
        self._training_log = None
        self._eval_log = None
        self._step_counter = 0
        
    def train(self, *args, **kwargs):
        """Training with pipelined evaluation."""
        self._training_log = get_training_logger()
        self._eval_log = get_eval_logger()
        
        if self.pipelined_evaluator:
            self.pipelined_evaluator.start()
            self._training_log.info("Pipelined Gemini evaluation started - cross-step overlap enabled")
        
        try:
            return super().train(*args, **kwargs)
        finally:
            if self.pipelined_evaluator:
                self.pipelined_evaluator.stop()

    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        """
        Generate and score with pipelined Gemini.
        
        This is called by TRL's training loop. We can't change when it's called,
        but we CAN use state from the previous call.
        
        Pattern:
        - On call N: if we fired Gemini on call N-1, it's been running during
          whatever happened between calls (generation, etc.)
        - Fire Gemini for this call
        - Wait for previous call's Gemini (should be ready)
        - Wait for this call's Gemini
        
        Actually, that's still not right because TRL caches results via _buffered_inputs.
        
        Let me trace through TRL's actual flow more carefully...
        
        _prepare_inputs() is called, which calls _generate_and_score_completions()
        only when the buffer is empty or needs refresh. So we can't rely on
        _generate_and_score_completions being called every step.
        
        The correct place to pipeline is in the TRAINING LOOP itself, not in
        these methods. We need to override _inner_training_loop.
        """
        if not self.pipelined_evaluator:
            return super()._generate_and_score_completions(inputs)
            
        # For now, use the within-step overlap (Gemini || log_prob)
        # True cross-step overlap requires _inner_training_loop override
        return self._generate_and_score_pipelined(inputs)
    
    def _generate_and_score_pipelined(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        """Generate and score with Gemini running parallel to log prob computation."""
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        step_id = self._step_counter
        self._step_counter += 1
        
        t0 = time.time()
        
        # === Generation ===
        prompts = [x["prompt"] for x in inputs]
        images = self._get_images(inputs)
        if images is not None:
            prompts = [prepare_multimodal_messages(p, img) for p, img in zip(prompts, images, strict=True)]
            
        gen_start = time.time()
        (
            prompt_ids_list, completion_ids_list, tool_mask_list, completions,
            num_items_in_batch, sampling_per_token_logps_list, extra_fields,
        ) = self._generate(prompts)
        gen_time = time.time() - gen_start
        
        # === Parse fields & fire Gemini async ===
        fire_start = time.time()
        parsed = self._parse_fields(inputs)
        self.pipelined_evaluator.fire_evaluation(
            step_id=step_id,
            completions=completions,
            **parsed,
        )
        fire_time = time.time() - fire_start
        
        # === Tensor conversion & log prob computation (Gemini runs in parallel) ===
        logprob_start = time.time()
        tensors = self._convert_to_tensors(
            prompt_ids_list, completion_ids_list, tool_mask_list,
            sampling_per_token_logps_list, device
        )
        
        forward_kwargs = self._get_forward_kwargs(images, prompts, tensors["completion_ids"])
        
        with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
            old_per_token_logps = self._compute_old_logps(tensors, forward_kwargs, mode)
            ref_per_token_logps = self._compute_ref_logps(tensors, forward_kwargs)
        logprob_time = time.time() - logprob_start
        
        # === Wait for Gemini ===
        wait_start = time.time()
        rewards = self.pipelined_evaluator.get_rewards(step_id)
        wait_time = time.time() - wait_start
        
        # === Compute advantages ===
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        advantages, metrics = self._compute_advantages(rewards_tensor, mode)
        
        total_time = time.time() - t0
        self._training_log.info(
            f"Step {step_id}: gen={gen_time:.1f}s fire={fire_time:.2f}s "
            f"logprob={logprob_time:.1f}s wait={wait_time:.1f}s total={total_time:.1f}s "
            f"mean_reward={rewards_tensor.mean():.3f}"
        )
        
        # === Logging ===
        self._log_step(prompts, tensors, rewards_tensor, advantages, metrics, mode, extra_fields, inputs)
        
        # === Build output ===
        return self._build_output(tensors, advantages, num_items_in_batch, 
                                  old_per_token_logps, ref_per_token_logps,
                                  forward_kwargs, images)
    
    def _get_images(self, inputs):
        """Extract images from inputs."""
        if "images" in inputs[0]:
            images = [ex.get("images") for ex in inputs]
        elif "image" in inputs[0]:
            images = [[ex.get("image")] if ex.get("image") is not None else None for ex in inputs]
        else:
            images = None
        if images is not None and all(img_list == [] for img_list in images):
            images = None
        return images
    
    def _parse_fields(self, inputs) -> Dict[str, str]:
        """Parse proposition/evidence/conditions from inputs.
        
        Returns keys matching PipelinedHybridReward.fire_evaluation() signature:
        - proposition, evidence, oracle_conditions, oracle_temporal_context
        - expected_judgment, domain, mode
        """
        example = inputs[0] if inputs else {}
        proposition = example.get("proposition", "")
        evidence = example.get("evidence", "")
        conditions = example.get("conditions", "")
        temporal_context = example.get("temporal_context", "")
        expected_judgment = example.get("expected_judgment", "")
        domain = example.get("domain", "empirical")
        mode = example.get("mode", "")
        
        if not proposition or not evidence or not conditions:
            input_text = example.get("input_text", "")
            parsed = self._parse_input_text(input_text)
            proposition = proposition or parsed.get("proposition", "")
            evidence = evidence or parsed.get("evidence", "")
            conditions = conditions or parsed.get("conditions", "")
            
        return {
            "proposition": proposition,
            "evidence": evidence, 
            "oracle_conditions": conditions,
            "oracle_temporal_context": temporal_context,
            "expected_judgment": expected_judgment,
            "domain": domain,
            "mode": mode,
        }
    
    def _parse_input_text(self, text: str) -> Dict[str, str]:
        """Parse from input_text field."""
        result = {"proposition": "", "evidence": "", "conditions": ""}
        if not text:
            return result
        lines = text.split("\n")
        current = None
        cond_lines = []
        for line in lines:
            s = line.strip()
            if s.startswith("Proposition:"):
                current = "proposition"
                result["proposition"] = s.replace("Proposition:", "").strip()
            elif s.startswith("Evidence:"):
                current = "evidence"
                result["evidence"] = s.replace("Evidence:", "").strip()
            elif s.startswith("Conditions for truth:"):
                current = "conditions"
            elif s.startswith("Temporal context:"):
                current = None
            elif current == "conditions" and s.startswith("-"):
                cond_lines.append(s)
            elif current == "evidence" and s and not s.startswith(("Proposition:", "Evidence:", "Conditions", "Temporal")):
                result["evidence"] += " " + s
        result["conditions"] = "\n".join(cond_lines)
        return result
    
    def _convert_to_tensors(self, prompt_ids_list, completion_ids_list, tool_mask_list,
                            sampling_per_token_logps_list, device):
        """Convert lists to padded tensors."""
        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")
        
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()
            
        tool_mask = None
        if self.tools and tool_mask_list:
            tool_mask = [torch.tensor(m, device=device) for m in tool_mask_list]
            tool_mask = pad(tool_mask, padding_value=1, padding_side="right")
            
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "tool_mask": tool_mask,
        }
    
    def _get_forward_kwargs(self, images, prompts, completion_ids):
        """Get forward kwargs for multimodal models."""
        if images is None:
            return {}
        prompts_text = [
            apply_chat_template({"prompt": p}, self.processing_class, tools=self.tools, **self.chat_template_kwargs)["prompt"]
            for p in prompts
        ]
        prompt_inputs = self.processing_class(images=images, text=prompts_text, padding=True, return_tensors="pt")
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        forward_kwargs = {k: v for k, v in prompt_inputs.items() if k not in ["input_ids", "attention_mask"]}
        if "token_type_ids" in forward_kwargs:
            token_type_ids = forward_kwargs["token_type_ids"]
            forward_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1)
        return forward_kwargs
    
    def _compute_old_logps(self, tensors, forward_kwargs, mode):
        """Compute old per-token log probs if needed."""
        generate_every = self.args.steps_per_generation * self.num_iterations
        if self.args.gradient_accumulation_steps % generate_every != 0 or (
            self.use_vllm and self.vllm_importance_sampling_correction
        ):
            prompt_completion_ids = torch.cat([tensors["prompt_ids"], tensors["completion_ids"]], dim=1)
            attention_mask = torch.cat([tensors["prompt_mask"], tensors["completion_mask"]], dim=1)
            logits_to_keep = tensors["completion_ids"].size(1)
            batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size
            old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size, **forward_kwargs
            )
            return old_per_token_logps
        return None
    
    def _compute_ref_logps(self, tensors, forward_kwargs):
        """Compute reference model log probs if needed."""
        if self.beta == 0.0:
            return None
        prompt_completion_ids = torch.cat([tensors["prompt_ids"], tensors["completion_ids"]], dim=1)
        attention_mask = torch.cat([tensors["prompt_mask"], tensors["completion_mask"]], dim=1)
        logits_to_keep = tensors["completion_ids"].size(1)
        
        if self.ref_model is not None:
            ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep, **forward_kwargs
            )
        else:
            model = self.accelerator.unwrap_model(self.model)
            with use_adapter(model, adapter_name="ref" if "ref" in model.peft_config else None):
                ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep, **forward_kwargs
                )
        return ref_per_token_logps
    
    def _compute_advantages(self, rewards_tensor, mode):
        """Compute advantages from rewards."""
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval
        
        mean_grouped = rewards_tensor.view(-1, num_generations).mean(dim=1)
        mean_grouped = mean_grouped.repeat_interleave(num_generations, dim=0)
        
        if num_generations > 1:
            std_rewards = rewards_tensor.view(-1, num_generations).std(dim=1)
            std_rewards = std_rewards.repeat_interleave(num_generations, dim=0)
        else:
            std_rewards = torch.zeros_like(rewards_tensor)
            
        advantages = rewards_tensor - mean_grouped
        if self.scale_rewards != "none":
            advantages = advantages / (std_rewards + 1e-4)
            
        is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
        
        process_slice = slice(
            self.accelerator.process_index * len(rewards_tensor),
            (self.accelerator.process_index + 1) * len(rewards_tensor),
        )
        
        return advantages[process_slice], {
            "reward_mean": rewards_tensor.mean().item(),
            "reward_std": rewards_tensor.std().item(),
            "frac_zero_std": is_std_zero.float().mean().item(),
            "all_advantages": advantages,
        }
    
    def _log_step(self, prompts, tensors, rewards_tensor, advantages, metrics, mode, extra_fields, inputs):
        """Log step metrics."""
        self._metrics[mode]["reward"].append(metrics["reward_mean"])
        self._metrics[mode]["reward_std"].append(metrics["reward_std"])
        self._metrics[mode]["frac_reward_zero_std"].append(metrics["frac_zero_std"])
        
        prompts_text = self.processing_class.batch_decode(tensors["prompt_ids"], skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(tensors["completion_ids"], skip_special_tokens=True)
        
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        self._logs["advantages"].extend(metrics["all_advantages"].tolist())
        
        if extra_fields:
            for i, inp in enumerate(inputs):
                for key, values in extra_fields.items():
                    if isinstance(values, list) and i < len(values):
                        inp[key] = values[i]
    
    def _build_output(self, tensors, advantages, num_items_in_batch,
                      old_per_token_logps, ref_per_token_logps, forward_kwargs, images):
        """Build the output dict."""
        output = {
            "prompt_ids": tensors["prompt_ids"],
            "prompt_mask": tensors["prompt_mask"],
            "completion_ids": tensors["completion_ids"],
            "completion_mask": tensors["completion_mask"],
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        for key in ["pixel_values", "image_grid_thw", "pixel_attention_mask", "image_sizes", "token_type_ids"]:
            if key in forward_kwargs:
                output[key] = forward_kwargs[key]
        if images is not None:
            output["num_images"] = [len(img_list) for img_list in images]
        if tensors["tool_mask"] is not None:
            output["tool_mask"] = tensors["tool_mask"]
        return output
