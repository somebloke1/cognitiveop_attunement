"""
Async Pipelined Reward Evaluation for GRPO Training.

This module provides async infrastructure to overlap Gemini API latency with
model generation, reducing per-step time from ~68s to ~10s.

The Problem:
    TRL's GRPO training loop is sequential:
        for step in steps:
            completions = generate()      # ~5s  (GPU)
            rewards = reward_fn()         # ~60s (Gemini blocking)
            update_weights()              # ~3s  (GPU)
        Per-step: ~68s, bottleneck is Gemini

The Solution:
    Overlap Gemini call for step N with generation for step N+1:
        fire_gemini(step=0, data=...)     # async, returns immediately
        for step in steps:
            completions = generate()      # ~5s, Gemini running in parallel
            rewards = wait_gemini(step)   # ~0s if ready, else blocks
            fire_gemini(step+1, data=...) # async for next step
            update_weights()              # ~3s
        Per-step: max(8s, gemini_latency) ≈ 10s if Gemini < 60s

Key Design:
    - `AsyncGeminiEvaluator`: Low-level async wrapper around LlmEvaluator
    - `PipelinedHybridReward`: High-level reward function with async support
    - Reuses all existing LlmEvaluator logic (prompt building, parsing, caching)
    - Thread-safe: async runs in background thread, main thread is sync

Usage:
    from src.evaluation.async_reward import PipelinedHybridReward
    from src.evaluation.llm_evaluator import LlmEvaluator
    
    llm_eval = LlmEvaluator(use_cache=True)
    llm_eval.create_cache()  # Important: create cache before starting
    
    reward = PipelinedHybridReward(llm_eval)
    reward.start()
    
    # Fire async (returns immediately after surface analysis)
    surface = reward.fire_evaluation(step_id=0, completions=..., ...)
    
    # ... do other work (model generation for next step) ...
    
    # Get rewards (blocks if not ready)
    rewards = reward.get_rewards(step_id=0)
    
    reward.stop()
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from threading import Thread, Lock
from typing import Optional, List, Dict, Any, Tuple

from google.genai import types

from .surface_analyzer import SurfaceAnalyzer, SurfaceAnalysis
from .llm_evaluator import (
    LlmEvaluator,
    SemanticEvaluation,
    EvaluationResponse,
    BatchEvaluationResponse,
    BATCH_EVALUATION_PROMPT,
    EVALUATION_PROMPT_ENHANCED,
    EVALUATOR_SYSTEM_INSTRUCTION,
    _extract_gemini_metadata,
)
from .logging_config import get_eval_logger, get_gemini_logger

# Import InferenceRecord for structured logging (optional - may not be available in all contexts)
try:
    from src.logging import InferenceRecord
    HAS_INFERENCE_RECORD = True
except ImportError:
    HAS_INFERENCE_RECORD = False


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class AsyncRequest:
    """A pending async Gemini request."""
    step_id: int
    fired_at: float
    future: Future
    # Data needed for reward computation after response arrives
    completions: List[str]
    surface_analyses: List[SurfaceAnalysis]
    expected_judgment: str


@dataclass  
class AsyncResult:
    """Result from an async Gemini evaluation."""
    step_id: int
    semantic_evals: List[SemanticEvaluation]
    elapsed_seconds: float
    success: bool
    error: Optional[str] = None


# =============================================================================
# AsyncGeminiEvaluator - Low-level async Gemini client
# =============================================================================


class AsyncGeminiEvaluator:
    """
    Manages async Gemini API calls in a background thread.
    
    This wraps LlmEvaluator to provide non-blocking API calls. It reuses
    all the existing logic (prompt building, caching, parsing) and only
    replaces the sync API call with an async one.
    
    Thread safety: All public methods are thread-safe.
    """
    
    def __init__(self, llm_evaluator: LlmEvaluator, max_concurrent: int = 8, run_id: str = ""):
        """
        Initialize with an existing LlmEvaluator.
        
        Args:
            llm_evaluator: Configured evaluator (handles cache, parsing, etc.)
                           IMPORTANT: Call create_cache() before start() if caching desired.
            max_concurrent: Maximum concurrent Gemini API calls (default 8)
            run_id: Unique run identifier for logging
        """
        self.llm_evaluator = llm_evaluator
        self.max_concurrent = max_concurrent
        self._current_concurrent = max_concurrent  # Dynamic, may decrease on 429
        self._run_id = run_id
        
        # Async infrastructure
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[Thread] = None
        self._running = False
        self._semaphore: Optional[asyncio.Semaphore] = None
        
        # Rate limit backoff state
        self._rate_limit_backoff_until: float = 0.0  # Timestamp when backoff ends
        self._rate_limit_lock = Lock()
        
        # Request tracking
        self._pending: Dict[int, AsyncRequest] = {}
        self._results: Dict[int, AsyncResult] = {}
        self._lock = Lock()
        
        # Logging
        self._gemini_log = None
        self._eval_log = None
        self._run_logger = None
    
    @property
    def run_id(self) -> str:
        return self._run_id
    
    @run_id.setter  
    def run_id(self, value: str) -> None:
        self._run_id = value
    
    def set_run_logger(self, run_logger) -> None:
        """Set run logger for structured inference tracking."""
        self._run_logger = run_logger
        
    def start(self) -> None:
        """Start the background event loop thread."""
        if self._running:
            return
            
        self._gemini_log = get_gemini_logger()
        self._eval_log = get_eval_logger()
        self._running = True
        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=self._run_loop, daemon=True, name="AsyncGemini")
        self._thread.start()
        
        self._current_concurrent = self.max_concurrent  # Reset on start
        self._semaphore = asyncio.Semaphore(self._current_concurrent)
        self._gemini_log.info("=== ASYNC GEMINI EVALUATOR THREAD STARTED ===")
        self._gemini_log.info(f"Ready to process evaluation requests (max_concurrent={self._current_concurrent})")
        self._gemini_log.info(f"Rate limit backoff: reduce concurrency by 1, wait 30s on 429")
        
    def stop(self) -> None:
        """Stop the background thread and cleanup."""
        if not self._running:
            return
            
        self._running = False
        
        # Stop the event loop
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            
        # Wait for thread to finish
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            
        if self._gemini_log:
            self._gemini_log.info("AsyncGeminiEvaluator stopped")
    
    async def _handle_rate_limit(self, inference_id: str) -> None:
        """
        Handle rate limit (429) error by reducing concurrency and backing off.
        
        - Reduces _current_concurrent by 1 (minimum 1)
        - Sets backoff period of 30 seconds
        - Waits for backoff before returning
        """
        with self._rate_limit_lock:
            old_concurrent = self._current_concurrent
            if self._current_concurrent > 1:
                self._current_concurrent -= 1
            
            # Set backoff until 30 seconds from now
            self._rate_limit_backoff_until = time.time() + 30.0
            
            self._gemini_log.warning(
                f"[inf_id={inference_id}] RATE_LIMITED: reducing concurrency {old_concurrent} -> {self._current_concurrent}, "
                f"backing off 30s"
            )
        
        # Wait for backoff period
        await asyncio.sleep(30.0)
        self._gemini_log.info(f"[inf_id={inference_id}] Rate limit backoff complete, resuming")
        
    def _run_loop(self) -> None:
        """Background thread entry point - runs the event loop."""
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            # Cleanup pending tasks
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            self._loop.close()
            
    def fire_request(
        self,
        step_id: int,
        user_prompt: str,
        completions: List[str],
        surface_analyses: List[SurfaceAnalysis],
        expected_judgment: str,
        individual_prompts: List[str] = None,
    ) -> None:
        """
        Fire an async Gemini request. Returns immediately.
        
        Args:
            step_id: Unique identifier (typically training step number)
            user_prompt: Formatted batch evaluation prompt for Gemini (legacy)
            completions: Model completions being evaluated
            surface_analyses: Pre-computed local analyses
            expected_judgment: Oracle-verified correct judgment (shared for batch)
            individual_prompts: If provided, evaluate in parallel with independent calls
                               (avoids batch homogenization)
        """
        if not self._running:
            raise RuntimeError("AsyncGeminiEvaluator not started - call start() first")
            
        # Schedule the async call - use parallel if individual prompts provided
        if individual_prompts:
            future = asyncio.run_coroutine_threadsafe(
                self._async_call_parallel(step_id, individual_prompts, surface_analyses, expected_judgment),
                self._loop
            )
            self._gemini_log.info(f"[run={self.run_id}][step={step_id}] PARALLEL REQUEST FIRED: n={len(individual_prompts)}, max_concurrent={self.max_concurrent}, expected={expected_judgment}")
        else:
            future = asyncio.run_coroutine_threadsafe(
                self._async_call(step_id, user_prompt, surface_analyses, expected_judgment),
                self._loop
            )
            self._gemini_log.info(f"[step={step_id}] BATCH REQUEST FIRED: completions={len(completions)}, expected={expected_judgment}")
            self._gemini_log.debug(f"[step={step_id}] PROMPT ({len(user_prompt)} chars):\n{'='*60}\n{user_prompt}\n{'='*60}")
        
        request = AsyncRequest(
            step_id=step_id,
            fired_at=time.time(),
            future=future,
            completions=completions,
            surface_analyses=surface_analyses,
            expected_judgment=expected_judgment,
        )
        
        with self._lock:
            self._pending[step_id] = request
        
    async def _async_call(
        self,
        step_id: int,
        user_prompt: str,
        surface_analyses: List[SurfaceAnalysis],
        expected_judgment: str,
    ) -> None:
        """Async coroutine that calls Gemini and stores result."""
        start = time.time()
        n = len(surface_analyses)
        
        self._gemini_log.debug(f"[step={step_id}] ASYNC_CALL starting, n={n} completions")
        
        try:
            # Build config - same as sync _call_llm_batch
            config_params = {
                "temperature": self.llm_evaluator.temperature,
                "top_p": self.llm_evaluator.top_p,
                "top_k": self.llm_evaluator.top_k,
                "max_output_tokens": 16384,
                "response_mime_type": "application/json",
                "response_schema": BatchEvaluationResponse,
            }
            
            # Use cached system instruction if available, otherwise send full instruction
            if self.llm_evaluator._cache_name:
                config_params["cached_content"] = self.llm_evaluator._cache_name
                self._gemini_log.debug(f"[step={step_id}] Using cached content: {self.llm_evaluator._cache_name}")
            else:
                config_params["system_instruction"] = EVALUATOR_SYSTEM_INSTRUCTION
                self._gemini_log.debug(f"[step={step_id}] No cache - sending full system instruction ({len(EVALUATOR_SYSTEM_INSTRUCTION)} chars)")
                
            self._gemini_log.debug(f"[step={step_id}] Calling Gemini API (model={self.llm_evaluator.model_name})...")
                
            # Async API call
            response = await self.llm_evaluator.client.aio.models.generate_content(
                model=self.llm_evaluator.model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(**config_params),
            )
            
            elapsed = time.time() - start
            
            # Log metadata
            metadata = _extract_gemini_metadata(response)
            self._gemini_log.info(
                f"[step={step_id}] RESPONSE: elapsed={elapsed:.1f}s {metadata.to_log_string()}"
            )
            
            # Warn if potentially truncated
            if metadata.output_token_count > 0 and metadata.output_token_count < 700:
                self._gemini_log.warning(
                    f"[step={step_id}] Potentially truncated: only {metadata.output_token_count} output tokens"
                )
            
            # Parse response using existing method
            raw_response = response.text or ""
            
            # Log full response
            self._gemini_log.debug(
                f"[step={step_id}] RAW_RESPONSE ({len(raw_response)} chars):\n{'='*60}\n{raw_response}\n{'='*60}"
            )
            
            semantic_evals = self.llm_evaluator._parse_batch_response(
                raw_response, expected_judgment, surface_analyses, metadata.finish_reason
            )
            
            # Log parsed evaluations
            for i, sem_eval in enumerate(semantic_evals):
                self._gemini_log.debug(
                    f"[step={step_id}] EVAL[{i}]: holistic={sem_eval.holistic_score:.3f} "
                    f"correct={sem_eval.judgment_correct} fallback={sem_eval.is_fallback}"
                )
            
            result = AsyncResult(
                step_id=step_id,
                semantic_evals=semantic_evals,
                elapsed_seconds=elapsed,
                success=True,
            )
            
            self._gemini_log.info(f"[step={step_id}] SUCCESS: {n} evaluations parsed in {elapsed:.1f}s")
            self._gemini_log.info(f"[step={step_id}] ASYNC PROCESSING READY - rewards available")
            
        except Exception as e:
            elapsed = time.time() - start
            self._gemini_log.error(f"[step={step_id}] FAILED after {elapsed:.1f}s: {type(e).__name__}: {e}")
            import traceback
            self._gemini_log.debug(f"[step={step_id}] TRACEBACK:\n{traceback.format_exc()}")
            
            # Create fallback evaluations using existing method
            semantic_evals = [
                self.llm_evaluator._create_fallback_evaluation(
                    surface, expected_judgment, str(e)
                )
                for surface in surface_analyses
            ]
            
            result = AsyncResult(
                step_id=step_id,
                semantic_evals=semantic_evals,
                elapsed_seconds=elapsed,
                success=False,
                error=str(e),
            )
            
        # Store result and cleanup pending
        with self._lock:
            self._results[step_id] = result
            self._pending.pop(step_id, None)
            self._gemini_log.debug(f"[step={step_id}] Result stored, pending count={len(self._pending)}")
    
    async def _async_call_parallel(
        self,
        step_id: int,
        individual_prompts: List[str],
        surface_analyses: List[SurfaceAnalysis],
        expected_judgment: str,
    ) -> None:
        """
        Evaluate completions in PARALLEL with independent Gemini calls.
        
        This avoids the batch homogenization problem where Gemini anchors
        all scores similarly when evaluating multiple completions together.
        """
        start = time.time()
        n = len(individual_prompts)
        
        self._gemini_log.debug(f"[step={step_id}] PARALLEL_CALL starting, n={n} independent evals")
        
        # Config for single evaluation
        config_params = {
            "temperature": self.llm_evaluator.temperature,
            "top_p": self.llm_evaluator.top_p,
            "top_k": self.llm_evaluator.top_k,
            "max_output_tokens": 16384,  # Increased to handle schema overhead + full JSON responses
            "response_mime_type": "application/json",
            "response_schema": EvaluationResponse,  # Single, not batch
        }
        
        if self.llm_evaluator._cache_name:
            config_params["cached_content"] = self.llm_evaluator._cache_name
        
        async def eval_single(idx: int, prompt: str) -> tuple:
            """Evaluate single completion, return (index, SemanticEvaluation)."""
            # Unique inference_id: run.step.R.idx (R=Remote/Gemini)
            # Short run suffix for log readability (last 6 chars of run_id)
            run_suffix = self.run_id[-6:] if self.run_id else "norun"
            inference_id = f"{run_suffix}.{step_id}.R.{idx}"
            
            async with self._semaphore:
                inf_start = time.time()
                self._gemini_log.debug(f"[inf_id={inference_id}] ACQUIRED semaphore, starting API call")
                
                try:
                    response = await self.llm_evaluator.client.aio.models.generate_content(
                        model=self.llm_evaluator.model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(**config_params),
                    )
                    inf_elapsed = time.time() - inf_start
                    
                    raw = response.text or ""
                    
                    # Extract finish reason and response metadata
                    finish_reason = None
                    response_id = None
                    if response.candidates and len(response.candidates) > 0:
                        finish_reason = response.candidates[0].finish_reason
                        if hasattr(response, 'response_id'):
                            response_id = response.response_id
                    
                    # Log call completion with full details
                    self._gemini_log.info(
                        f"[inf_id={inference_id}] API_COMPLETE: {inf_elapsed:.1f}s, "
                        f"finish={finish_reason}, len={len(raw)}, resp_id={response_id}"
                    )
                    
                    if str(finish_reason) != "FinishReason.STOP":
                        self._gemini_log.warning(
                            f"[inf_id={inference_id}] NON-STOP FINISH: {finish_reason}, response may be truncated"
                        )
                    
                    eval_result = self.llm_evaluator._parse_single_response(
                        raw, expected_judgment, surface_analyses[idx]
                    )
                    
                    self._gemini_log.debug(
                        f"[inf_id={inference_id}] PARSED: correct={eval_result.judgment_correct}, "
                        f"scores=[cond={eval_result.condition_identification_score:.2f}, "
                        f"ev={eval_result.evidence_mapping_score:.2f}, "
                        f"rev={eval_result.reversion_score:.2f}]"
                    )
                    
                    # Log structured inference record for remote call
                    if self._run_logger and HAS_INFERENCE_RECORD:
                        from datetime import datetime
                        
                        # Save prompt and response content
                        content_path = self._run_logger.save_content(
                            inference_id=inference_id,
                            content_type="full",
                            content=f"=== PROMPT ===\n{prompt}\n\n=== RESPONSE ===\n{raw}",
                        )
                        
                        record = InferenceRecord(
                            run_id=self.run_id,
                            step_id=step_id,
                            inference_id=inference_id,
                            inference_type="remote",
                            inference_idx=idx,
                            start_time=datetime.fromtimestamp(inf_start).isoformat(),
                            end_time=datetime.now().isoformat(),
                            duration_s=inf_elapsed,
                            status="completed",
                            finish_reason=str(finish_reason) if finish_reason else None,
                            response_len=len(raw),
                            content_path=content_path,
                            response_id=response_id,
                            judgment_correct=eval_result.judgment_correct,
                            scores={
                                "condition_identification": eval_result.condition_identification_score,
                                "evidence_mapping": eval_result.evidence_mapping_score,
                                "reasoning_validity": eval_result.reasoning_validity_score,
                                "judgment_coherence": eval_result.judgment_coherence_score,
                                "operational_fidelity": eval_result.operational_fidelity_score,
                                "reversion": eval_result.reversion_score,
                                "authentic_intent": eval_result.authentic_intent_score,
                                "conciseness": eval_result.conciseness_score,
                            },
                            is_fallback=False,
                        )
                        self._run_logger.log_inference(record)
                    
                    return (idx, eval_result)
                    
                except Exception as e:
                    inf_elapsed = time.time() - inf_start
                    error_str = str(e)
                    
                    # Check for rate limit error (429)
                    is_rate_limit = (
                        "429" in error_str or 
                        "RESOURCE_EXHAUSTED" in error_str or
                        "ResourceExhausted" in type(e).__name__ or
                        "TooManyRequests" in type(e).__name__ or
                        "rate limit" in error_str.lower()
                    )
                    
                    if is_rate_limit:
                        await self._handle_rate_limit(inference_id)
                    
                    self._gemini_log.error(
                        f"[inf_id={inference_id}] API_FAILED after {inf_elapsed:.1f}s: {type(e).__name__}: {e}"
                        + (" [RATE_LIMITED]" if is_rate_limit else "")
                    )
                    fallback = self.llm_evaluator._create_fallback_evaluation(
                        surface_analyses[idx], expected_judgment, str(e)
                    )
                    
                    # Log failed inference record
                    if self._run_logger and HAS_INFERENCE_RECORD:
                        from datetime import datetime
                        record = InferenceRecord(
                            run_id=self.run_id,
                            step_id=step_id,
                            inference_id=inference_id,
                            inference_type="remote",
                            inference_idx=idx,
                            start_time=datetime.fromtimestamp(inf_start).isoformat(),
                            end_time=datetime.now().isoformat(),
                            duration_s=inf_elapsed,
                            status="rate_limited" if is_rate_limit else "failed",
                            error=f"{type(e).__name__}: {e}",
                            is_fallback=True,
                        )
                        self._run_logger.log_inference(record)
                    
                    return (idx, fallback)
        
        try:
            # Fire all evaluations in parallel
            tasks = [eval_single(i, p) for i, p in enumerate(individual_prompts)]
            results = await asyncio.gather(*tasks)
            
            # Sort by index to maintain order
            results.sort(key=lambda x: x[0])
            semantic_evals = [r[1] for r in results]
            
            elapsed = time.time() - start
            
            # Log results
            correct_count = sum(1 for e in semantic_evals if e.judgment_correct)
            # Calculate timing stats
            call_times = [r[1].eval_time if hasattr(r[1], 'eval_time') else 0 for r in results]
            self._gemini_log.info(
                f"[step={step_id}] PARALLEL COMPLETE: {correct_count}/{n} correct, "
                f"wall_time={elapsed:.1f}s, calls={n}"
            )
            
            # Log batch summary record
            if self._run_logger and HAS_INFERENCE_RECORD:
                from datetime import datetime
                batch_record = InferenceRecord(
                    run_id=self.run_id,
                    step_id=step_id,
                    inference_id=f"{self.run_id[-6:]}.{step_id}.R.BATCH",
                    inference_type="remote_batch_summary",
                    inference_idx=-1,  # -1 indicates batch summary
                    start_time=datetime.fromtimestamp(start).isoformat(),
                    end_time=datetime.now().isoformat(),
                    duration_s=elapsed,
                    status="completed",
                    token_count=n,  # Using token_count to store batch size
                    judgment_correct=correct_count == n,  # All correct?
                    scores={"correct_count": correct_count, "total_count": n, "success_rate": correct_count / n},
                )
                self._run_logger.log_inference(batch_record)
            
            result = AsyncResult(
                step_id=step_id,
                semantic_evals=semantic_evals,
                elapsed_seconds=elapsed,
                success=True,
            )
            
        except Exception as e:
            elapsed = time.time() - start
            self._gemini_log.error(f"[step={step_id}] PARALLEL FAILED after {elapsed:.1f}s: {e}")
            
            semantic_evals = [
                self.llm_evaluator._create_fallback_evaluation(
                    surface, expected_judgment, str(e)
                )
                for surface in surface_analyses
            ]
            
            result = AsyncResult(
                step_id=step_id,
                semantic_evals=semantic_evals,
                elapsed_seconds=elapsed,
                success=False,
                error=str(e),
            )
        
        # Store result
        with self._lock:
            self._results[step_id] = result
            self._pending.pop(step_id, None)
            self._gemini_log.debug(f"[step={step_id}] Parallel result stored")
            
    def get_result(self, step_id: int, timeout: float = 120.0) -> AsyncResult:
        """
        Get result for a fired request. Blocks until ready or timeout.
        
        Args:
            step_id: Step ID to get result for
            timeout: Maximum seconds to wait
            
        Returns:
            AsyncResult with semantic evaluations
            
        Raises:
            KeyError: If step_id was never fired
            TimeoutError: If result not ready within timeout
        """
        start = time.time()
        
        self._gemini_log.debug(f"[step={step_id}] GET_RESULT called")
        
        # Check if already available
        with self._lock:
            if step_id in self._results:
                result = self._results.pop(step_id)
                self._gemini_log.info(
                    f"[step={step_id}] RESULT READY immediately (was cached), "
                    f"success={result.success}, elapsed_api={result.elapsed_seconds:.1f}s"
                )
                return result
            if step_id not in self._pending:
                raise KeyError(f"Step {step_id} was not fired")
            request = self._pending[step_id]
            time_in_flight = time.time() - request.fired_at
            self._gemini_log.debug(f"[step={step_id}] Waiting for result (in flight for {time_in_flight:.1f}s)...")
            
        # Wait for future to complete
        try:
            request.future.result(timeout=timeout)
        except asyncio.TimeoutError:
            self._gemini_log.error(f"[step={step_id}] TIMEOUT after {timeout}s")
            raise TimeoutError(f"Timeout waiting for step {step_id}")
        except Exception as e:
            # Future raised an exception - result should still be stored
            self._gemini_log.warning(f"[step={step_id}] Future exception: {e}")
            
        wait_time = time.time() - start
        
        # Now should be in results
        with self._lock:
            if step_id in self._results:
                result = self._results.pop(step_id)
                self._gemini_log.info(
                    f"[step={step_id}] RESULT RETRIEVED after {wait_time:.1f}s wait, "
                    f"success={result.success}, api_time={result.elapsed_seconds:.1f}s"
                )
                return result
                
        raise RuntimeError(f"Step {step_id} future completed but no result stored")
        
    def is_ready(self, step_id: int) -> bool:
        """Check if result is ready without blocking."""
        with self._lock:
            return step_id in self._results
            
    def pending_count(self) -> int:
        """Number of requests in flight."""
        with self._lock:
            return len(self._pending)


# =============================================================================
# PipelinedHybridReward - High-level reward function with async support
# =============================================================================


class PipelinedHybridReward:
    """
    Reward function that supports pipelined async Gemini evaluation.
    
    This wraps LlmEvaluator with async capability and provides the same
    interface as HybridRewardFunction, plus fire-and-get methods for
    pipelined training.
    
    Two modes of operation:
    
    1. Sync mode (via compute_rewards_sync):
       Same as HybridRewardFunction - fire and wait in one call.
       
    2. Async mode (via fire_evaluation + get_rewards):
       Fire request, do other work, then get results.
    """
    
    def __init__(
        self,
        llm_evaluator: LlmEvaluator,
        surface_analyzer: Optional[SurfaceAnalyzer] = None,
        correctness_weight: float = 0.3,
        semantic_weight: float = 0.7,
        correct_reward: float = 1.0,
        incorrect_reward: float = -0.5,
        no_judgment_reward: float = -0.3,
        max_concurrent: int = 8,
        run_id: str = "",
    ):
        """
        Initialize pipelined reward function.
        
        Args:
            llm_evaluator: Configured LLM evaluator (call create_cache before start)
            surface_analyzer: Local surface analyzer (created if None)
            correctness_weight: Weight for binary correctness (default 0.3)
            semantic_weight: Weight for semantic quality (default 0.7)
            correct_reward: Reward for correct judgment
            incorrect_reward: Reward for incorrect judgment
            no_judgment_reward: Reward for missing judgment
            max_concurrent: Max parallel Gemini API calls (default 8, dynamic backoff on 429)
            run_id: Unique run identifier for logging
        """
        self.llm_evaluator = llm_evaluator
        self.surface_analyzer = surface_analyzer or SurfaceAnalyzer()
        self.max_concurrent = max_concurrent
        self._run_id = run_id
        
        self.correctness_weight = correctness_weight
        self.semantic_weight = semantic_weight
        self.correct_reward = correct_reward
        self.incorrect_reward = incorrect_reward
        self.no_judgment_reward = no_judgment_reward
        
        # Async evaluator (created on start())
        self._async_eval: Optional[AsyncGeminiEvaluator] = None
        
        # Run logger reference (set externally by trainer)
        self._run_logger = None
        
        # Cached data for reward computation
        self._step_data: Dict[int, Dict[str, Any]] = {}
        self._lock = Lock()
        
        # Logging
        self._eval_log = None
    
    @property
    def run_id(self) -> str:
        return self._run_id
    
    @run_id.setter
    def run_id(self, value: str) -> None:
        self._run_id = value
        if self._async_eval:
            self._async_eval.run_id = value
        
    def start(self) -> None:
        """Start async infrastructure. Must be called before fire_evaluation."""
        if self._async_eval is not None:
            return
            
        self._eval_log = get_eval_logger()
        self._async_eval = AsyncGeminiEvaluator(
            self.llm_evaluator, 
            max_concurrent=self.max_concurrent,
            run_id=self.run_id
        )
        self._async_eval.start()
        
        self._eval_log.logger.info("PipelinedHybridReward started")
        
    def set_run_logger(self, run_logger) -> None:
        """Set the run logger for structured inference tracking."""
        self._run_logger = run_logger
        if self._async_eval:
            self._async_eval.set_run_logger(run_logger)
    
    def stop(self) -> None:
        """Stop async infrastructure. Call after training completes."""
        if self._async_eval:
            self._async_eval.stop()
            self._async_eval = None
            
        if self._eval_log:
            self._eval_log.logger.info("PipelinedHybridReward stopped")
            
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
        
    def run_surface_analysis(
        self,
        completions: List[str],
        domain: str,
    ) -> List[SurfaceAnalysis]:
        """
        Run local surface analysis (fast, sync).
        
        Args:
            completions: Model completions
            domain: Domain for all completions (shared in GRPO)
            
        Returns:
            List of surface analyses
        """
        return [
            self.surface_analyzer.analyze(comp, domain)
            for comp in completions
        ]
        
    def _build_batch_prompt(
        self,
        completions: List[str],
        proposition: str,
        evidence: str,
        oracle_conditions: str,
        oracle_temporal_context: str,
        expected_judgment: str,
        domain: str,
        surface_analyses: List[SurfaceAnalysis],
        mode: str = "",
        enhanced_context: str = "",
    ) -> str:
        """Build Gemini batch evaluation prompt (DEPRECATED - use individual prompts)."""
        n = len(completions)
        
        # Build completions section with numbered entries
        completions_section_parts = []
        for i, (comp, surface) in enumerate(zip(completions, surface_analyses)):
            completions_section_parts.append(
                f"### Completion {i + 1}\n\n{comp}\n\n"
                f"**Surface Analysis {i + 1}**:\n{surface.to_prompt_section()}"
            )
        completions_section = "\n\n---\n\n".join(completions_section_parts)
        
        return BATCH_EVALUATION_PROMPT.format(
            num_completions=n,
            domain=domain,
            mode=mode or "not specified",
            proposition=proposition,
            evidence=evidence,
            oracle_conditions=oracle_conditions or "(not specified)",
            oracle_temporal_context=oracle_temporal_context or "(not specified)",
            expected_judgment=expected_judgment,
            enhanced_context=enhanced_context or "",
            completions_section=completions_section,
        )
    
    def _build_individual_prompts(
        self,
        completions: List[str],
        proposition: str,
        evidence: str,
        oracle_conditions: str,
        expected_judgment: str,
        domain: str,
        surface_analyses: List[SurfaceAnalysis],
        mode: str = "",
        enhanced_context: str = "",
    ) -> List[str]:
        """Build individual evaluation prompts for parallel processing."""
        prompts = []
        for comp, surface in zip(completions, surface_analyses):
            prompt = EVALUATION_PROMPT_ENHANCED.format(
                domain=domain,
                mode=mode or "not specified",
                proposition=proposition,
                evidence=evidence,
                conditions=oracle_conditions or "(not specified)",
                expected_judgment=expected_judgment,
                completion=comp,
                surface_analysis=surface.to_prompt_section(),
                enhanced_context=enhanced_context or "",
            )
            prompts.append(prompt)
        return prompts
        
    def fire_evaluation(
        self,
        step_id: int,
        completions: List[str],
        proposition: str,
        evidence: str,
        oracle_conditions: str,
        oracle_temporal_context: str,
        expected_judgment: str,
        domain: str,
        mode: str = "",
        enhanced_context: str = "",
    ) -> List[SurfaceAnalysis]:
        """
        Fire async Gemini evaluation. Returns surface analyses immediately.
        
        This does three things:
        1. Runs local surface analysis (sync, ~1s)
        2. Builds the Gemini prompt
        3. Fires the async request (returns immediately)
        
        Call get_rewards(step_id) later to get the final rewards.
        
        Args:
            step_id: Unique step identifier
            completions: Model completions to evaluate
            proposition: The proposition being judged
            evidence: Evidence text
            oracle_conditions: Oracle conditions for grading
            oracle_temporal_context: Oracle temporal context for grading
            expected_judgment: Oracle's correct judgment
            domain: Domain type
            mode: Fulfillment mode
            enhanced_context: Additional context for evaluation
            
        Returns:
            List of surface analyses (for immediate use if needed)
        """
        if self._async_eval is None:
            raise RuntimeError("PipelinedHybridReward not started - call start() first")
            
        # Run surface analysis locally (fast)
        surface_analyses = self.run_surface_analysis(completions, domain)
        
        # Build INDIVIDUAL prompts for parallel evaluation (avoids batch homogenization)
        individual_prompts = self._build_individual_prompts(
            completions=completions,
            proposition=proposition,
            evidence=evidence,
            oracle_conditions=oracle_conditions,
            expected_judgment=expected_judgment,
            domain=domain,
            surface_analyses=surface_analyses,
            mode=mode,
            enhanced_context=enhanced_context,
        )
        
        # Store data for later reward computation
        with self._lock:
            self._step_data[step_id] = {
                "completions": completions,
                "surface_analyses": surface_analyses,
                "expected_judgment": expected_judgment,
            }
            
        # Fire async request with individual prompts (parallel independent calls)
        self._async_eval.fire_request(
            step_id=step_id,
            user_prompt="",  # Not used when individual_prompts provided
            completions=completions,
            surface_analyses=surface_analyses,
            expected_judgment=expected_judgment,
            individual_prompts=individual_prompts,  # NEW: enables parallel mode
        )
        
        return surface_analyses
        
    def get_rewards(self, step_id: int, timeout: float = 120.0) -> List[float]:
        """
        Get rewards for a previously fired evaluation. Blocks until ready.
        
        Args:
            step_id: Step ID from fire_evaluation()
            timeout: Maximum seconds to wait
            
        Returns:
            List of reward values (one per completion)
        """
        if self._async_eval is None:
            raise RuntimeError("PipelinedHybridReward not started")
            
        # Get cached step data
        with self._lock:
            step_data = self._step_data.pop(step_id, None)
            
        if step_data is None:
            raise KeyError(f"Step {step_id} was not fired or already retrieved")
            
        # Wait for Gemini result
        result = self._async_eval.get_result(step_id, timeout=timeout)
        
        # Check for fallbacks - if any eval is fallback, discard entire batch
        # Fallback evals have zero variance (identical scores) → zero gradient → wasted step
        has_fallback = any(sem.is_fallback for sem in result.semantic_evals)
        if has_fallback:
            if self._eval_log:
                self._eval_log.logger.warning(
                    f"Step {step_id}: Discarding batch due to Gemini parse failure (fallback evals)"
                )
            return None  # Signal to skip this step
        
        # Compute rewards using Gemini's judgment_correct (no surface analyzer needed)
        rewards = []
        for semantic in result.semantic_evals:
            reward = self._compute_reward(semantic)
            rewards.append(reward)
            
        # Log batch summary
        if self._eval_log:
            import statistics
            mean_r = statistics.mean(rewards) if rewards else 0.0
            std_r = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
            correct_count = sum(1 for sem in result.semantic_evals if sem.judgment_correct)
            self._eval_log.log_batch_summary(
                len(rewards), rewards, mean_r, std_r,
                correct_count=correct_count,
                failure_count=len(rewards) - correct_count,
            )
            
        return rewards
        
    def is_ready(self, step_id: int) -> bool:
        """Check if evaluation result is ready without blocking."""
        if self._async_eval is None:
            return False
        return self._async_eval.is_ready(step_id)
        
    def _compute_reward(
        self,
        semantic: SemanticEvaluation,
    ) -> float:
        """Compute reward from Gemini's semantic evaluation.
        
        Uses Gemini's judgment_correct (not surface extraction) because:
        - Gemini can interpret varied judgment formats (quotes, markdown, etc.)
        - Surface regex extraction is brittle and introduces systematic errors
        - Fallback batches are now discarded, so we always have valid semantic evals
        """
        # Correctness from Gemini's assessment
        correctness = self.correct_reward if semantic.judgment_correct else self.incorrect_reward
            
        # Composite reward
        reward = (
            self.correctness_weight * correctness
            + self.semantic_weight * semantic.holistic_score
        )
        
        return reward
        
    def compute_rewards_sync(
        self,
        completions: List[str],
        proposition: str,
        evidence: str,
        conditions: str,
        expected_judgment: str,
        domain: str,
        mode: str = "",
    ) -> List[float]:
        """
        Compute rewards synchronously (blocking).
        
        This is equivalent to fire_evaluation + get_rewards in sequence.
        Use this for non-pipelined training or the first/last step.
        """
        # Use a sentinel step_id for sync calls
        step_id = -1
        
        self.fire_evaluation(
            step_id=step_id,
            completions=completions,
            proposition=proposition,
            evidence=evidence,
            conditions=conditions,
            expected_judgment=expected_judgment,
            domain=domain,
            mode=mode,
        )
        return self.get_rewards(step_id)
