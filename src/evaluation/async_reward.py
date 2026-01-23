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
    BatchEvaluationResponse,
    BATCH_EVALUATION_PROMPT,
    _extract_gemini_metadata,
)
from .logging_config import get_eval_logger, get_gemini_logger


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
    
    def __init__(self, llm_evaluator: LlmEvaluator):
        """
        Initialize with an existing LlmEvaluator.
        
        Args:
            llm_evaluator: Configured evaluator (handles cache, parsing, etc.)
                           IMPORTANT: Call create_cache() before start() if caching desired.
        """
        self.llm_evaluator = llm_evaluator
        
        # Async infrastructure
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[Thread] = None
        self._running = False
        
        # Request tracking
        self._pending: Dict[int, AsyncRequest] = {}
        self._results: Dict[int, AsyncResult] = {}
        self._lock = Lock()
        
        # Logging
        self._gemini_log = None
        self._eval_log = None
        
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
        
        self._gemini_log.info("=== ASYNC GEMINI EVALUATOR THREAD STARTED ===")
        self._gemini_log.info("Ready to process evaluation requests")
        
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
    ) -> None:
        """
        Fire an async Gemini request. Returns immediately.
        
        Args:
            step_id: Unique identifier (typically training step number)
            user_prompt: Formatted batch evaluation prompt for Gemini
            completions: Model completions being evaluated
            surface_analyses: Pre-computed local analyses
            expected_judgment: Oracle-verified correct judgment (shared for batch)
        """
        if not self._running:
            raise RuntimeError("AsyncGeminiEvaluator not started - call start() first")
            
        # Schedule the async call
        future = asyncio.run_coroutine_threadsafe(
            self._async_call(step_id, user_prompt, surface_analyses, expected_judgment),
            self._loop
        )
        
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
        
        # Log request details
        self._gemini_log.info(f"[step={step_id}] REQUEST FIRED: completions={len(completions)}, expected={expected_judgment}")
        self._gemini_log.debug(f"[step={step_id}] PROMPT ({len(user_prompt)} chars):\n{'='*60}\n{user_prompt}\n{'='*60}")
        
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
            
            # Use cached system instruction if available
            if self.llm_evaluator._cache_name:
                config_params["cached_content"] = self.llm_evaluator._cache_name
                self._gemini_log.debug(f"[step={step_id}] Using cached content: {self.llm_evaluator._cache_name}")
            else:
                self._gemini_log.debug(f"[step={step_id}] No cache - sending full system instruction")
                
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
                raw_response, expected_judgment, surface_analyses
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
        """
        self.llm_evaluator = llm_evaluator
        self.surface_analyzer = surface_analyzer or SurfaceAnalyzer()
        
        self.correctness_weight = correctness_weight
        self.semantic_weight = semantic_weight
        self.correct_reward = correct_reward
        self.incorrect_reward = incorrect_reward
        self.no_judgment_reward = no_judgment_reward
        
        # Async evaluator (created on start())
        self._async_eval: Optional[AsyncGeminiEvaluator] = None
        
        # Cached data for reward computation
        self._step_data: Dict[int, Dict[str, Any]] = {}
        self._lock = Lock()
        
        # Logging
        self._eval_log = None
        
    def start(self) -> None:
        """Start async infrastructure. Must be called before fire_evaluation."""
        if self._async_eval is not None:
            return
            
        self._eval_log = get_eval_logger()
        self._async_eval = AsyncGeminiEvaluator(self.llm_evaluator)
        self._async_eval.start()
        
        self._eval_log.logger.info("PipelinedHybridReward started")
        
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
        """Build Gemini batch evaluation prompt (same format as sync version)."""
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
        
        # Build prompt
        prompt = self._build_batch_prompt(
            completions=completions,
            proposition=proposition,
            evidence=evidence,
            oracle_conditions=oracle_conditions,
            oracle_temporal_context=oracle_temporal_context,
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
            
        # Fire async request
        self._async_eval.fire_request(
            step_id=step_id,
            user_prompt=prompt,
            completions=completions,
            surface_analyses=surface_analyses,
            expected_judgment=expected_judgment,
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
        
        # Compute rewards
        rewards = []
        expected = step_data["expected_judgment"]
        
        for surface, semantic in zip(
            step_data["surface_analyses"],
            result.semantic_evals,
        ):
            reward = self._compute_reward(surface, semantic, expected)
            rewards.append(reward)
            
        # Log batch summary
        if self._eval_log:
            import statistics
            mean_r = statistics.mean(rewards) if rewards else 0.0
            std_r = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
            self._eval_log.log_batch_summary(
                len(rewards), rewards, mean_r, std_r,
                correct_count=sum(1 for r in rewards if r > 0.5),
                failure_count=sum(1 for r in rewards if r <= 0.5),
            )
            
        return rewards
        
    def is_ready(self, step_id: int) -> bool:
        """Check if evaluation result is ready without blocking."""
        if self._async_eval is None:
            return False
        return self._async_eval.is_ready(step_id)
        
    def _compute_reward(
        self,
        surface: SurfaceAnalysis,
        semantic: SemanticEvaluation,
        expected_judgment: str,
    ) -> float:
        """Compute single reward from surface + semantic analysis."""
        # Correctness component
        if surface.judgment_extracted is None:
            correctness = self.no_judgment_reward
        elif surface.judgment_extracted.lower() == expected_judgment.lower():
            correctness = self.correct_reward
            # Reduce for fallback evaluations (lower confidence)
            if semantic.is_fallback:
                correctness *= 0.5
        else:
            correctness = self.incorrect_reward
            
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
