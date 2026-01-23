"""
Tests for async reward evaluation infrastructure.

These tests verify:
1. AsyncGeminiEvaluator lifecycle (start/stop)
2. PipelinedHybridReward fire/get pattern
3. Thread safety and error handling
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import asyncio

from src.evaluation.async_reward import (
    AsyncGeminiEvaluator,
    PipelinedHybridReward,
    AsyncRequest,
    AsyncResult,
)
from src.evaluation.surface_analyzer import SurfaceAnalysis
from src.evaluation.llm_evaluator import SemanticEvaluation


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_llm_evaluator():
    """Create a mock LlmEvaluator."""
    evaluator = Mock()
    evaluator.temperature = 0.1
    evaluator.top_p = 0.95
    evaluator.top_k = 20
    evaluator.model_name = "test-model"
    evaluator._cache_name = None
    
    # Mock the client
    evaluator.client = Mock()
    evaluator.client.aio = Mock()
    evaluator.client.aio.models = Mock()
    
    # Mock _parse_batch_response to return valid evaluations
    def mock_parse(raw_response, expected_judgment, surface_analyses):
        return [
            SemanticEvaluation(
                condition_identification_score=0.5,
                evidence_mapping_score=0.5,
                reasoning_validity_score=0.5,
                judgment_coherence_score=0.5,
                operational_fidelity_score=0.5,
                authentic_intent_score=0.5,
                judgment_correct=True,
                holistic_score=0.5,
            )
            for _ in surface_analyses
        ]
    evaluator._parse_batch_response = Mock(side_effect=mock_parse)
    
    # Mock _create_fallback_evaluation
    def mock_fallback(surface, expected, error):
        return SemanticEvaluation(
            condition_identification_score=0.3,
            evidence_mapping_score=0.3,
            reasoning_validity_score=0.3,
            judgment_coherence_score=0.3,
            operational_fidelity_score=0.3,
            authentic_intent_score=0.3,
            judgment_correct=False,
            holistic_score=0.3,
            is_fallback=True,
        )
    evaluator._create_fallback_evaluation = Mock(side_effect=mock_fallback)
    
    return evaluator


@pytest.fixture
def mock_surface_analyses():
    """Create mock surface analyses."""
    return [
        SurfaceAnalysis(
            judgment_extracted="Yes",
            has_condition_language=True,
            has_fulfillment_language=True,
        ),
        SurfaceAnalysis(
            judgment_extracted="No",
            has_condition_language=True,
            has_fulfillment_language=False,
        ),
    ]


# =============================================================================
# AsyncGeminiEvaluator Tests
# =============================================================================


class TestAsyncGeminiEvaluator:
    """Tests for AsyncGeminiEvaluator."""
    
    def test_start_stop_lifecycle(self, mock_llm_evaluator):
        """Test that evaluator can start and stop cleanly."""
        evaluator = AsyncGeminiEvaluator(mock_llm_evaluator)
        
        assert not evaluator._running
        assert evaluator._thread is None
        assert evaluator._loop is None
        
        evaluator.start()
        
        assert evaluator._running
        assert evaluator._thread is not None
        assert evaluator._thread.is_alive()
        assert evaluator._loop is not None
        
        evaluator.stop()
        
        assert not evaluator._running
        
    def test_start_idempotent(self, mock_llm_evaluator):
        """Test that calling start() twice is safe."""
        evaluator = AsyncGeminiEvaluator(mock_llm_evaluator)
        
        evaluator.start()
        thread1 = evaluator._thread
        
        evaluator.start()  # Should be no-op
        thread2 = evaluator._thread
        
        assert thread1 is thread2
        
        evaluator.stop()
        
    def test_stop_without_start(self, mock_llm_evaluator):
        """Test that stop() without start() is safe."""
        evaluator = AsyncGeminiEvaluator(mock_llm_evaluator)
        evaluator.stop()  # Should not raise
        
    def test_fire_request_without_start_raises(self, mock_llm_evaluator, mock_surface_analyses):
        """Test that fire_request raises if not started."""
        evaluator = AsyncGeminiEvaluator(mock_llm_evaluator)
        
        with pytest.raises(RuntimeError, match="not started"):
            evaluator.fire_request(
                step_id=0,
                user_prompt="test prompt",
                completions=["test completion"],
                surface_analyses=mock_surface_analyses[:1],
                expected_judgment="Yes",
            )
            
    def test_get_result_unknown_step_raises(self, mock_llm_evaluator):
        """Test that get_result raises for unknown step."""
        evaluator = AsyncGeminiEvaluator(mock_llm_evaluator)
        evaluator.start()
        
        try:
            with pytest.raises(KeyError, match="was not fired"):
                evaluator.get_result(step_id=999)
        finally:
            evaluator.stop()
            
    def test_pending_count(self, mock_llm_evaluator):
        """Test pending count tracking."""
        evaluator = AsyncGeminiEvaluator(mock_llm_evaluator)
        evaluator.start()
        
        try:
            assert evaluator.pending_count() == 0
        finally:
            evaluator.stop()


# =============================================================================
# PipelinedHybridReward Tests
# =============================================================================


class TestPipelinedHybridReward:
    """Tests for PipelinedHybridReward."""
    
    def test_start_stop_lifecycle(self, mock_llm_evaluator):
        """Test that reward function can start and stop cleanly."""
        reward = PipelinedHybridReward(mock_llm_evaluator)
        
        assert reward._async_eval is None
        
        reward.start()
        
        assert reward._async_eval is not None
        
        reward.stop()
        
        assert reward._async_eval is None
        
    def test_context_manager(self, mock_llm_evaluator):
        """Test context manager usage."""
        reward = PipelinedHybridReward(mock_llm_evaluator)
        
        with reward:
            assert reward._async_eval is not None
            
        assert reward._async_eval is None
        
    def test_run_surface_analysis(self, mock_llm_evaluator):
        """Test surface analysis runs correctly."""
        reward = PipelinedHybridReward(mock_llm_evaluator)
        
        completions = ["JUDGMENT: Yes\nREASONING: Test"]
        domain = "empirical"
        
        analyses = reward.run_surface_analysis(completions, domain)
        
        assert len(analyses) == 1
        assert analyses[0].judgment_extracted == "Yes"
        
    def test_fire_evaluation_without_start_raises(self, mock_llm_evaluator):
        """Test that fire_evaluation raises if not started."""
        reward = PipelinedHybridReward(mock_llm_evaluator)
        
        with pytest.raises(RuntimeError, match="not started"):
            reward.fire_evaluation(
                step_id=0,
                completions=["test"],
                proposition="test prop",
                evidence="test evidence",
                oracle_conditions="- test condition",
                oracle_temporal_context="- Evidence timeframe: test",
                expected_judgment="Yes",
                domain="empirical",
            )
            
    def test_get_rewards_without_fire_raises(self, mock_llm_evaluator):
        """Test that get_rewards raises for unfired step."""
        reward = PipelinedHybridReward(mock_llm_evaluator)
        reward.start()
        
        try:
            with pytest.raises(KeyError, match="was not fired"):
                reward.get_rewards(step_id=999)
        finally:
            reward.stop()
            
    def test_reward_computation(self, mock_llm_evaluator):
        """Test reward computation formula."""
        reward = PipelinedHybridReward(
            mock_llm_evaluator,
            correctness_weight=0.3,
            semantic_weight=0.7,
            correct_reward=1.0,
            incorrect_reward=-0.5,
        )
        
        # Correct judgment
        surface = SurfaceAnalysis(judgment_extracted="Yes")
        semantic = SemanticEvaluation(holistic_score=0.6, is_fallback=False)
        
        r = reward._compute_reward(surface, semantic, "Yes")
        expected = 0.3 * 1.0 + 0.7 * 0.6  # 0.3 + 0.42 = 0.72
        assert abs(r - expected) < 0.001
        
        # Incorrect judgment
        r = reward._compute_reward(surface, semantic, "No")
        expected = 0.3 * (-0.5) + 0.7 * 0.6  # -0.15 + 0.42 = 0.27
        assert abs(r - expected) < 0.001
        
    def test_reward_computation_fallback_penalty(self, mock_llm_evaluator):
        """Test that fallback evaluations get reduced correctness."""
        reward = PipelinedHybridReward(
            mock_llm_evaluator,
            correctness_weight=0.3,
            semantic_weight=0.7,
            correct_reward=1.0,
        )
        
        surface = SurfaceAnalysis(judgment_extracted="Yes")
        semantic = SemanticEvaluation(holistic_score=0.6, is_fallback=True)
        
        r = reward._compute_reward(surface, semantic, "Yes")
        # Correctness reduced by 50% for fallback: 1.0 * 0.5 = 0.5
        expected = 0.3 * 0.5 + 0.7 * 0.6  # 0.15 + 0.42 = 0.57
        assert abs(r - expected) < 0.001


# =============================================================================
# Integration-style Tests (with mocked Gemini)
# =============================================================================


class TestAsyncIntegration:
    """Integration tests with mocked Gemini responses."""
    
    @pytest.fixture
    def mock_gemini_response(self):
        """Create a mock Gemini response."""
        response = Mock()
        response.text = '{"evaluations": [{"condition_identification_score": 0.6}]}'
        
        # Add usage metadata
        response.usage_metadata = Mock()
        response.usage_metadata.prompt_token_count = 1000
        response.usage_metadata.cached_content_token_count = 500
        response.usage_metadata.candidates_token_count = 200
        
        # Add candidates
        candidate = Mock()
        candidate.finish_reason = "STOP"
        candidate.finish_message = None
        candidate.safety_ratings = []
        response.candidates = [candidate]
        
        return response
    
    def test_fire_and_get_with_mocked_gemini(
        self, mock_llm_evaluator, mock_gemini_response
    ):
        """Test complete fire/get cycle with mocked Gemini."""
        # Set up async mock
        async def mock_generate_content(*args, **kwargs):
            await asyncio.sleep(0.01)  # Simulate small latency
            return mock_gemini_response
            
        mock_llm_evaluator.client.aio.models.generate_content = AsyncMock(
            side_effect=mock_generate_content
        )
        
        reward = PipelinedHybridReward(mock_llm_evaluator)
        reward.start()
        
        try:
            # Fire evaluation
            surface = reward.fire_evaluation(
                step_id=0,
                completions=["JUDGMENT: Yes\nREASONING: Test"],
                proposition="Test proposition",
                evidence="Test evidence",
                oracle_conditions="- Test condition",
                oracle_temporal_context="- Evidence timeframe: test",
                expected_judgment="Yes",
                domain="empirical",
            )
            
            assert len(surface) == 1
            
            # Get rewards (should succeed)
            rewards = reward.get_rewards(step_id=0, timeout=5.0)
            
            assert len(rewards) == 1
            assert isinstance(rewards[0], float)
            
        finally:
            reward.stop()


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_completions(self, mock_llm_evaluator):
        """Test handling of empty completions list."""
        reward = PipelinedHybridReward(mock_llm_evaluator)
        
        analyses = reward.run_surface_analysis([], "empirical")
        assert analyses == []
        
    def test_no_judgment_reward(self, mock_llm_evaluator):
        """Test reward when no judgment extracted."""
        reward = PipelinedHybridReward(
            mock_llm_evaluator,
            correctness_weight=0.3,
            semantic_weight=0.7,
            no_judgment_reward=-0.3,
        )
        
        surface = SurfaceAnalysis(judgment_extracted=None)  # No judgment
        semantic = SemanticEvaluation(holistic_score=0.4, is_fallback=False)
        
        r = reward._compute_reward(surface, semantic, "Yes")
        expected = 0.3 * (-0.3) + 0.7 * 0.4  # -0.09 + 0.28 = 0.19
        assert abs(r - expected) < 0.001
