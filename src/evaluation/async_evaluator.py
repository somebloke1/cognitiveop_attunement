"""
Async Prefetch Evaluator: Overlaps Gemini API calls with local computation.

This module provides an async wrapper around LlmEvaluator that allows
firing Gemini requests before they're needed, hiding latency by
overlapping with local model generation and surface analysis.

Architecture:
    - Main training loop fires prefetch request for step N+1
    - While waiting for step N rewards, step N+1 request is in flight
    - When step N+1 needs rewards, response is (usually) already available

This reduces effective per-step time from:
    local_gen + surface_analysis + gemini_latency + weight_update
To:
    max(local_gen + surface_analysis + weight_update, gemini_latency)

Since gemini_latency (~30-80s) >> local work (~5-10s), this is significant.
"""

import asyncio
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Callable
from threading import Thread, Lock
import queue

from google import genai
from google.genai import types

from .llm_evaluator import (
    LlmEvaluator,
    SemanticEvaluation,
    EvaluationResponse,
    BatchEvaluationResponse,
    EVALUATOR_SYSTEM_INSTRUCTION,
    EVALUATION_PROMPT_ENHANCED,
    _extract_gemini_metadata,
)
from .surface_analyzer import SurfaceAnalysis
from .logging_config import get_eval_logger, get_gemini_logger


@dataclass
class PrefetchRequest:
    """A request queued for async Gemini evaluation."""

    step_id: int
    user_prompt: str  # Legacy: batch prompt (deprecated)
    num_completions: int
    expected_judgment: str
    surface_analyses: List[SurfaceAnalysis]
    # NEW: Individual prompts for parallel evaluation
    individual_prompts: List[str] = None
    # Will be populated when response arrives
    future: asyncio.Future = None


@dataclass
class PrefetchResult:
    """Result from a prefetched Gemini evaluation."""

    step_id: int
    semantic_evals: List[SemanticEvaluation]
    success: bool
    error: Optional[str] = None


class AsyncPrefetchEvaluator:
    """
    Wraps LlmEvaluator to provide async prefetching of Gemini evaluations.

    Usage:
        evaluator = AsyncPrefetchEvaluator(llm_evaluator)
        evaluator.start()

        # In training loop:
        # Step N: fire prefetch for step N, get result for step N-1 (if any)
        evaluator.prefetch(step_id=n, prompt=..., ...)
        result = evaluator.get_result(step_id=n-1)  # blocks if not ready

        evaluator.stop()
    """

    def __init__(
        self,
        llm_evaluator: LlmEvaluator,
        max_in_flight: int = 2,
    ):
        """
        Initialize async prefetch evaluator.

        Args:
            llm_evaluator: Base evaluator to wrap
            max_in_flight: Maximum concurrent Gemini requests (default 2)
        """
        self.llm_evaluator = llm_evaluator
        self.max_in_flight = max_in_flight

        # Async infrastructure
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[Thread] = None
        self._running = False

        # Request/result queues
        self._pending_requests: Dict[int, PrefetchRequest] = {}
        self._results: Dict[int, PrefetchResult] = {}
        self._lock = Lock()

        # Async client for Gemini
        self._async_client = None

    def start(self):
        """Start the async event loop in a background thread."""
        if self._running:
            return

        self._running = True
        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        # Initialize async Gemini client
        self._async_client = self.llm_evaluator.client.aio

        get_eval_logger().logger.info("AsyncPrefetchEvaluator started")

    def stop(self):
        """Stop the async event loop and background thread."""
        if not self._running:
            return

        self._running = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5.0)

        get_eval_logger().logger.info("AsyncPrefetchEvaluator stopped")

    def _run_loop(self):
        """Run the event loop in background thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def prefetch(
        self,
        step_id: int,
        user_prompt: str,
        num_completions: int,
        expected_judgment: str,
        surface_analyses: List[SurfaceAnalysis],
    ) -> None:
        """
        Fire an async Gemini request for future retrieval.

        This returns immediately - the request runs in background.

        Args:
            step_id: Unique identifier for this request (usually step number)
            user_prompt: Formatted prompt for Gemini batch evaluation
            num_completions: Number of completions being evaluated
            expected_judgment: Expected judgment for correctness checking
            surface_analyses: Pre-computed surface analyses
        """
        if not self._running:
            raise RuntimeError("AsyncPrefetchEvaluator not started")

        request = PrefetchRequest(
            step_id=step_id,
            user_prompt=user_prompt,
            num_completions=num_completions,
            expected_judgment=expected_judgment,
            surface_analyses=surface_analyses,
        )

        with self._lock:
            self._pending_requests[step_id] = request

        # Schedule async task
        future = asyncio.run_coroutine_threadsafe(
            self._async_evaluate(request), self._loop
        )
        request.future = future

        get_eval_logger().logger.debug(f"Prefetch fired for step {step_id}")

    async def _async_evaluate(self, request: PrefetchRequest) -> None:
        """
        Async coroutine that evaluates completions in PARALLEL.
        
        Each completion gets its own independent Gemini call, ensuring
        truly independent scoring without batch homogenization.
        """
        gemini_log = get_gemini_logger()

        try:
            # Use individual prompts if available, otherwise fall back to batch
            if request.individual_prompts:
                semantic_evals = await self._evaluate_parallel(
                    request.step_id,
                    request.individual_prompts,
                    request.expected_judgment,
                    request.surface_analyses,
                )
            else:
                # Legacy batch path
                semantic_evals = await self._evaluate_batch(
                    request.step_id,
                    request.user_prompt,
                    request.expected_judgment,
                    request.surface_analyses,
                )

            result = PrefetchResult(
                step_id=request.step_id,
                semantic_evals=semantic_evals,
                success=True,
            )

        except Exception as e:
            gemini_log.error(
                f"Async evaluation failed step={request.step_id}: {e}"
            )

            # Create fallback evaluations
            semantic_evals = [
                self.llm_evaluator._create_fallback_evaluation(
                    surface, request.expected_judgment, str(e)
                )
                for surface in request.surface_analyses
            ]

            result = PrefetchResult(
                step_id=request.step_id,
                semantic_evals=semantic_evals,
                success=False,
                error=str(e),
            )

        # Store result
        with self._lock:
            self._results[request.step_id] = result
            if request.step_id in self._pending_requests:
                del self._pending_requests[request.step_id]

    async def _evaluate_parallel(
        self,
        step_id: int,
        individual_prompts: List[str],
        expected_judgment: str,
        surface_analyses: List[SurfaceAnalysis],
    ) -> List[SemanticEvaluation]:
        """
        Evaluate completions in parallel with independent Gemini calls.
        
        This ensures each completion gets truly independent scoring,
        avoiding the batch homogenization problem.
        """
        gemini_log = get_gemini_logger()
        n = len(individual_prompts)
        
        # Config for single evaluation (not batch)
        config_params = {
            "temperature": self.llm_evaluator.temperature,
            "top_p": self.llm_evaluator.top_p,
            "top_k": self.llm_evaluator.top_k,
            "max_output_tokens": 4096,  # Smaller for single eval
            "response_mime_type": "application/json",
            "response_schema": EvaluationResponse,  # Single, not batch
        }
        
        if self.llm_evaluator._cache_name:
            config_params["cached_content"] = self.llm_evaluator._cache_name
        
        async def eval_single(idx: int, prompt: str) -> tuple[int, SemanticEvaluation]:
            """Evaluate single completion, return (index, result)."""
            try:
                response = await self._async_client.models.generate_content(
                    model=self.llm_evaluator.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(**config_params),
                )
                
                raw = response.text or ""
                # Parse single evaluation response
                eval_result = self.llm_evaluator._parse_single_response(
                    raw, expected_judgment, surface_analyses[idx]
                )
                return (idx, eval_result)
                
            except Exception as e:
                gemini_log.warning(
                    f"Parallel eval {idx} failed step={step_id}: {e}"
                )
                fallback = self.llm_evaluator._create_fallback_evaluation(
                    surface_analyses[idx], expected_judgment, str(e)
                )
                return (idx, fallback)
        
        # Fire all evaluations in parallel
        tasks = [eval_single(i, p) for i, p in enumerate(individual_prompts)]
        results = await asyncio.gather(*tasks)
        
        # Sort by index to maintain order
        results.sort(key=lambda x: x[0])
        semantic_evals = [r[1] for r in results]
        
        gemini_log.debug(
            f"Parallel eval step={step_id}: {n} completions evaluated independently"
        )
        
        return semantic_evals

    async def _evaluate_batch(
        self,
        step_id: int,
        user_prompt: str,
        expected_judgment: str,
        surface_analyses: List[SurfaceAnalysis],
    ) -> List[SemanticEvaluation]:
        """
        Legacy batch evaluation (single call for all completions).
        
        DEPRECATED: Use _evaluate_parallel for independent scoring.
        """
        gemini_log = get_gemini_logger()
        
        config_params = {
            "temperature": self.llm_evaluator.temperature,
            "top_p": self.llm_evaluator.top_p,
            "top_k": self.llm_evaluator.top_k,
            "max_output_tokens": 16384,
            "response_mime_type": "application/json",
            "response_schema": BatchEvaluationResponse,
        }

        if self.llm_evaluator._cache_name:
            config_params["cached_content"] = self.llm_evaluator._cache_name

        response = await self._async_client.models.generate_content(
            model=self.llm_evaluator.model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(**config_params),
        )

        metadata = _extract_gemini_metadata(response)
        gemini_log.debug(
            f"Batch response step={step_id} {metadata.to_log_string()}"
        )

        raw_response = response.text or ""
        return self.llm_evaluator._parse_batch_response(
            raw_response,
            expected_judgment,
            surface_analyses,
            metadata.finish_reason,
        )

    def get_result(self, step_id: int, timeout: float = 120.0) -> PrefetchResult:
        """
        Get result for a previously prefetched request.

        Blocks until result is available or timeout expires.

        Args:
            step_id: Step ID to get result for
            timeout: Maximum seconds to wait

        Returns:
            PrefetchResult with semantic evaluations

        Raises:
            TimeoutError: If result not available within timeout
            KeyError: If step_id was never prefetched
        """
        # Check if already available
        with self._lock:
            if step_id in self._results:
                result = self._results.pop(step_id)
                return result
            if step_id not in self._pending_requests:
                raise KeyError(f"Step {step_id} was not prefetched")
            request = self._pending_requests[step_id]

        # Wait for future
        try:
            request.future.result(timeout=timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Timeout waiting for step {step_id} result")

        # Now should be in results
        with self._lock:
            if step_id in self._results:
                return self._results.pop(step_id)
            raise RuntimeError(f"Step {step_id} future completed but no result stored")

    def has_result(self, step_id: int) -> bool:
        """Check if result is available without blocking."""
        with self._lock:
            return step_id in self._results

    def pending_count(self) -> int:
        """Number of requests in flight."""
        with self._lock:
            return len(self._pending_requests)


class PipelinedRewardEvaluator:
    """
    High-level wrapper that manages pipelined reward computation.

    This integrates with the training loop to:
    1. Fire prefetch on each step
    2. Return rewards from previous step's prefetch
    3. Handle the first step specially (no previous prefetch)

    Usage:
        pipelined = PipelinedRewardEvaluator(hybrid_reward_fn)
        pipelined.start()

        for step in range(num_steps):
            # This fires prefetch for current step and returns
            # rewards from previous step (or None for step 0)
            rewards = pipelined.compute_rewards_pipelined(
                step_id=step,
                completions=completions,
                ...
            )

            if rewards is None:
                # First step - need to wait synchronously
                rewards = pipelined.flush_pending(step)

        pipelined.stop()
    """

    def __init__(self, base_evaluator: "HybridRewardFunction"):
        """
        Args:
            base_evaluator: The HybridRewardFunction to wrap
        """
        self.base_evaluator = base_evaluator
        self.async_evaluator: Optional[AsyncPrefetchEvaluator] = None
        self._current_step = -1
        self._pending_step_data: Dict[int, dict] = {}

    def start(self):
        """Initialize async infrastructure."""
        self.async_evaluator = AsyncPrefetchEvaluator(self.base_evaluator.llm_evaluator)
        self.async_evaluator.start()

    def stop(self):
        """Cleanup async infrastructure."""
        if self.async_evaluator:
            self.async_evaluator.stop()

    # Note: Full implementation would continue here with compute_rewards_pipelined
    # This is the foundation - integration with HybridRewardFunction would follow
