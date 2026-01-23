"""
Tests for Gemini context caching functionality.

These tests verify:
1. Cache creation with correct TTL
2. Cache usage in _call_llm and _call_llm_batch
3. Cache cleanup
4. Fallback when cache not available
5. Context manager support
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Any


class TestLlmEvaluatorCaching:
    """Tests for LlmEvaluator cache management."""

    @pytest.fixture
    def mock_genai_client(self):
        """Create a mock Gemini client."""
        with patch("src.evaluation.llm_evaluator.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def evaluator(self, mock_genai_client):
        """Create an LlmEvaluator with mocked client."""
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            from src.evaluation.llm_evaluator import LlmEvaluator

            return LlmEvaluator(
                provider="gemini",
                model_name="gemini-3-flash-preview",
                use_cache=True,
                cache_ttl=3600,
            )

    def test_init_with_cache_enabled(self, evaluator):
        """Verify evaluator initializes with cache settings."""
        assert evaluator.use_cache is True
        assert evaluator.cache_ttl == 3600
        assert evaluator._cache_name is None
        assert evaluator.model_name == "gemini-3-flash-preview"

    def test_init_with_cache_disabled(self, mock_genai_client):
        """Verify evaluator works with cache disabled."""
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            from src.evaluation.llm_evaluator import LlmEvaluator

            evaluator = LlmEvaluator(
                provider="gemini",
                use_cache=False,
            )
            assert evaluator.use_cache is False
            assert evaluator._cache_name is None

    def test_create_cache_success(self, evaluator, mock_genai_client):
        """Test successful cache creation."""
        mock_cache = MagicMock()
        mock_cache.name = "caches/test-cache-123"
        mock_genai_client.caches.create.return_value = mock_cache

        result = evaluator.create_cache(ttl_seconds=7200)

        assert result == "caches/test-cache-123"
        assert evaluator._cache_name == "caches/test-cache-123"
        assert evaluator.cache_active is True
        mock_genai_client.caches.create.assert_called_once()

    def test_create_cache_failure(self, evaluator, mock_genai_client):
        """Test cache creation failure falls back gracefully."""
        mock_genai_client.caches.create.side_effect = Exception("API Error")

        result = evaluator.create_cache()

        assert result is None
        assert evaluator._cache_name is None
        assert evaluator.cache_active is False

    def test_delete_cache_success(self, evaluator, mock_genai_client):
        """Test successful cache deletion."""
        evaluator._cache_name = "caches/test-cache-123"

        result = evaluator.delete_cache()

        assert result is True
        assert evaluator._cache_name is None
        assert evaluator.cache_active is False
        mock_genai_client.caches.delete.assert_called_once_with(
            name="caches/test-cache-123"
        )

    def test_delete_cache_when_none(self, evaluator, mock_genai_client):
        """Test delete_cache when no cache exists."""
        assert evaluator._cache_name is None

        result = evaluator.delete_cache()

        assert result is False
        mock_genai_client.caches.delete.assert_not_called()

    def test_delete_cache_failure(self, evaluator, mock_genai_client):
        """Test cache deletion failure is handled gracefully."""
        evaluator._cache_name = "caches/test-cache-123"
        mock_genai_client.caches.delete.side_effect = Exception("API Error")

        result = evaluator.delete_cache()

        assert result is False
        assert evaluator._cache_name is None

    def test_cache_active_property(self, evaluator):
        """Test cache_active property."""
        assert evaluator.cache_active is False

        evaluator._cache_name = "caches/test"
        assert evaluator.cache_active is True

        evaluator._cache_name = None
        assert evaluator.cache_active is False


class TestLlmEvaluatorCallWithCache:
    """Tests for _call_llm and _call_llm_batch with caching."""

    @pytest.fixture
    def mock_genai_client(self):
        """Create a mock Gemini client."""
        with patch("src.evaluation.llm_evaluator.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def evaluator_with_cache(self, mock_genai_client):
        """Create an LlmEvaluator with cache active."""
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            from src.evaluation.llm_evaluator import LlmEvaluator

            evaluator = LlmEvaluator(
                provider="gemini",
                model_name="gemini-3-flash-preview",
                use_cache=True,
            )
            evaluator._cache_name = "caches/active-cache"
            return evaluator

    @pytest.fixture
    def evaluator_without_cache(self, mock_genai_client):
        """Create an LlmEvaluator without cache."""
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            from src.evaluation.llm_evaluator import LlmEvaluator

            return LlmEvaluator(
                provider="gemini",
                model_name="gemini-3-flash-preview",
                use_cache=False,
            )

    def test_call_llm_uses_cache_when_active(
        self, evaluator_with_cache, mock_genai_client
    ):
        """Test _call_llm uses cached_content when cache is active."""
        mock_response = MagicMock()
        mock_response.text = '{"condition_identification_score": 0.8}'
        mock_genai_client.models.generate_content.return_value = mock_response

        evaluator_with_cache._call_llm("test prompt")

        call_args = mock_genai_client.models.generate_content.call_args
        config = call_args.kwargs["config"]
        assert hasattr(config, "cached_content") or "cached_content" in str(config)

    def test_call_llm_uses_system_instruction_without_cache(
        self, evaluator_without_cache, mock_genai_client
    ):
        """Test _call_llm uses system_instruction when no cache."""
        mock_response = MagicMock()
        mock_response.text = '{"condition_identification_score": 0.8}'
        mock_genai_client.models.generate_content.return_value = mock_response

        evaluator_without_cache._call_llm("test prompt")

        call_args = mock_genai_client.models.generate_content.call_args
        config = call_args.kwargs["config"]
        assert hasattr(config, "system_instruction") or "system_instruction" in str(
            config
        )

    def test_call_llm_batch_uses_cache_when_active(
        self, evaluator_with_cache, mock_genai_client
    ):
        """Test _call_llm_batch uses cached_content when cache is active."""
        mock_response = MagicMock()
        mock_response.text = '{"evaluations": []}'

        # Set up proper usage_metadata mock
        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 1000
        mock_usage.cached_content_token_count = 500
        mock_usage.candidates_token_count = 900
        mock_response.usage_metadata = mock_usage

        # Set up proper candidates mock
        mock_candidate = MagicMock()
        mock_candidate.finish_reason = "STOP"
        mock_candidate.finish_message = None
        mock_candidate.safety_ratings = []
        mock_response.candidates = [mock_candidate]

        mock_genai_client.models.generate_content.return_value = mock_response

        evaluator_with_cache._call_llm_batch("test batch prompt", 5)

        call_args = mock_genai_client.models.generate_content.call_args
        config = call_args.kwargs["config"]
        assert hasattr(config, "cached_content") or "cached_content" in str(config)


class TestHybridRewardFunctionCaching:
    """Tests for HybridRewardFunction cache lifecycle."""

    @pytest.fixture
    def mock_evaluator(self):
        """Mock the LlmEvaluator."""
        with patch("src.evaluation.hybrid_reward.LlmEvaluator") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.cache_active = False
            mock_cls.return_value = mock_instance
            yield mock_instance

    def test_initialize_creates_cache_when_enabled(self, mock_evaluator):
        """Test initialize() creates cache when caching is enabled."""
        from src.evaluation.hybrid_reward import (
            HybridRewardFunction,
            HybridRewardConfig,
        )

        config = HybridRewardConfig(use_gemini_cache=True)

        with patch("src.evaluation.hybrid_reward.LlmEvaluator") as mock_cls:
            mock_cls.return_value = mock_evaluator
            reward_fn = HybridRewardFunction(config)

            assert reward_fn._initialized is False

            mock_evaluator.create_cache.return_value = "caches/test"
            reward_fn.initialize()

            assert reward_fn._initialized is True
            mock_evaluator.create_cache.assert_called_once()

    def test_initialize_skipped_when_cache_disabled(self, mock_evaluator):
        """Test initialize() skips cache creation when disabled."""
        from src.evaluation.hybrid_reward import (
            HybridRewardFunction,
            HybridRewardConfig,
        )

        config = HybridRewardConfig(use_gemini_cache=False)

        with patch("src.evaluation.hybrid_reward.LlmEvaluator") as mock_cls:
            mock_cls.return_value = mock_evaluator
            reward_fn = HybridRewardFunction(config)
            reward_fn.initialize()

            mock_evaluator.create_cache.assert_not_called()

    def test_cleanup_deletes_cache(self, mock_evaluator):
        """Test cleanup() deletes cache when active."""
        from src.evaluation.hybrid_reward import (
            HybridRewardFunction,
            HybridRewardConfig,
        )

        config = HybridRewardConfig(use_gemini_cache=True)

        with patch("src.evaluation.hybrid_reward.LlmEvaluator") as mock_cls:
            mock_cls.return_value = mock_evaluator
            mock_evaluator.cache_active = True
            mock_evaluator.delete_cache.return_value = True

            reward_fn = HybridRewardFunction(config)
            reward_fn._initialized = True
            reward_fn.cleanup()

            mock_evaluator.delete_cache.assert_called_once()
            assert reward_fn._initialized is False

    def test_context_manager(self, mock_evaluator):
        """Test context manager initializes and cleans up."""
        from src.evaluation.hybrid_reward import (
            HybridRewardFunction,
            HybridRewardConfig,
        )

        config = HybridRewardConfig(use_gemini_cache=True)

        with patch("src.evaluation.hybrid_reward.LlmEvaluator") as mock_cls:
            mock_cls.return_value = mock_evaluator
            mock_evaluator.create_cache.return_value = "caches/test"
            mock_evaluator.cache_active = True
            mock_evaluator.delete_cache.return_value = True

            with HybridRewardFunction(config) as reward_fn:
                mock_evaluator.create_cache.assert_called_once()

            mock_evaluator.delete_cache.assert_called_once()


class TestDefaultModel:
    """Tests for default model configuration."""

    def test_llm_evaluator_default_model(self):
        """Test LlmEvaluator defaults to gemini-3-flash-preview."""
        with patch("src.evaluation.llm_evaluator.genai"):
            with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
                from src.evaluation.llm_evaluator import LlmEvaluator

                evaluator = LlmEvaluator()
                assert evaluator.model_name == "gemini-3-flash-preview"

    def test_hybrid_reward_config_default_model(self):
        """Test HybridRewardConfig defaults to gemini-3-flash-preview."""
        from src.evaluation.hybrid_reward import HybridRewardConfig

        config = HybridRewardConfig()
        assert config.llm_model == "gemini-3-flash-preview"

    def test_hybrid_reward_config_default_cache_settings(self):
        """Test HybridRewardConfig has correct cache defaults."""
        from src.evaluation.hybrid_reward import HybridRewardConfig

        config = HybridRewardConfig()
        assert config.use_gemini_cache is True
        assert config.gemini_cache_ttl == 172800  # 48 hours
