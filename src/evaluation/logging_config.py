"""
Logging configuration for the evaluation module.

Uses non-blocking QueueHandler for async-safe logging.
Each component gets its own queue and listener to ensure logs go to correct files.
"""

import logging
import logging.handlers
import os
import sys
import functools
import json
import queue
import atexit
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Any, Callable, Dict, Optional, List

# Valid log levels
LOG_LEVELS = {
    "error": logging.ERROR,
    "warn": logging.WARNING,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}

# Track listeners for cleanup
_queue_listeners: List[logging.handlers.QueueListener] = []

def _stop_all_listeners():
    """Stop all queue listeners on exit."""
    for listener in _queue_listeners:
        try:
            listener.stop()
        except:
            pass

atexit.register(_stop_all_listeners)


def setup_component_logger(
    component: str,
    level: Optional[str] = None,
    log_dir: str = "logs",
    timestamp: Optional[str] = None,
) -> logging.Logger:
    """
    Create a component-specific logger with its own log file.
    Uses QueueHandler for non-blocking async-safe logging.
    """
    file_level_str = (level or "debug").lower()
    file_level = LOG_LEVELS.get(file_level_str, logging.DEBUG)
    
    logger = logging.getLogger(f"cognitive_eval.{component}")
    logger.setLevel(file_level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{component}_{ts}.log"
    
    # Create actual file handler (runs in background thread)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    
    # Create dedicated queue for this component
    log_queue = queue.Queue(-1)
    
    # Use QueueHandler for non-blocking logging
    queue_handler = logging.handlers.QueueHandler(log_queue)
    queue_handler.setLevel(file_level)
    logger.addHandler(queue_handler)
    
    # Start dedicated listener for this component
    listener = logging.handlers.QueueListener(
        log_queue, file_handler, respect_handler_level=True
    )
    listener.start()
    _queue_listeners.append(listener)
    
    # Don't propagate to parent to keep streams separate
    logger.propagate = False
    
    logger.info(f"=== {component.upper()} LOGGER INITIALIZED (async) === (file: {log_file})")
    
    return logger


# Global timestamp for correlating component logs
_session_timestamp: Optional[str] = None


def get_session_timestamp() -> str:
    """Get or create session timestamp for correlating logs across components."""
    global _session_timestamp
    if _session_timestamp is None:
        _session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _session_timestamp


def setup_logging(
    level: Optional[str] = None,
    console_level: Optional[str] = None,
    log_dir: str = "logs",
    console: bool = True,
    file: bool = True,
    component_logs: bool = False,
) -> logging.Logger:
    """Configure logging for the evaluation module."""
    file_level_str = (level or os.getenv("LOG_LEVEL", "info")).lower()
    console_level_str = (console_level or os.getenv("CONSOLE_LOG_LEVEL", "warn")).lower()

    file_level = LOG_LEVELS.get(file_level_str, logging.INFO)
    console_level_num = LOG_LEVELS.get(console_level_str, logging.WARNING)
    min_level = min(file_level, console_level_num)

    logger = logging.getLogger("cognitive_eval")
    logger.setLevel(min_level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s.%(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level_num)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"evaluation_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.warning(f"Log file: {log_file}")

    logger.info(f"Logging initialized - console:{console_level_str.upper()} file:{file_level_str.upper()}")
    return logger


def get_logger() -> logging.Logger:
    """Get or create the evaluation logger."""
    logger = logging.getLogger("cognitive_eval")
    if not logger.handlers:
        setup_logging()
    return logger


# Component-specific logger cache
_component_loggers: Dict[str, logging.Logger] = {}


def get_component_logger(component: str) -> logging.Logger:
    """Get or create a component-specific logger."""
    if component not in _component_loggers:
        _component_loggers[component] = setup_component_logger(
            component, 
            timestamp=get_session_timestamp()
        )
    return _component_loggers[component]


def get_gemini_logger() -> logging.Logger:
    """Get logger for Gemini API operations."""
    return get_component_logger("gemini")


def get_training_logger() -> logging.Logger:
    """Get logger for training loop operations."""
    return get_component_logger("training")


def get_surface_logger() -> logging.Logger:
    """Get logger for surface analysis operations."""
    return get_component_logger("surface")


def truncate_for_log(text: str, max_length: int = 500) -> str:
    """Truncate text for logging, preserving start and end."""
    if len(text) <= max_length:
        return text
    half = max_length // 2 - 10
    return f"{text[:half]}\n... [{len(text) - max_length} chars truncated] ...\n{text[-half:]}"


def log_function_call(func: Callable) -> Callable:
    """Decorator to log function entry, exit, and arguments."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger()
        func_name = f"{func.__module__}.{func.__name__}"
        args_repr = [f"arg{i}=<{type(arg).__name__}>" for i, arg in enumerate(args)]
        kwargs_repr = [f"{k}=<{type(v).__name__}>" for k, v in kwargs.items()]
        logger.debug(f"ENTER {func_name}({', '.join(args_repr + kwargs_repr)})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"EXIT {func_name} -> <{type(result).__name__}>")
            return result
        except Exception as e:
            logger.error(f"ERROR {func_name} raised {type(e).__name__}: {e}")
            raise
    return wrapper


class EvaluationLogger:
    """Structured logging for evaluation operations."""

    def __init__(self):
        self.logger = get_logger()
        self._eval_count = 0

    def log_surface_analysis_input(self, completion: str, domain: str):
        self.logger.debug(f"SURFACE_INPUT domain={domain} completion_len={len(completion)}")

    def log_surface_analysis_output(self, analysis: Any):
        self.logger.debug(f"SURFACE_OUTPUT judgment={analysis.judgment_extracted}")

    def log_llm_request(self, provider: str, model: str, system_instruction: str, user_prompt: str):
        self._eval_count += 1
        self.logger.info(f"LLM_REQUEST #{self._eval_count} provider={provider} model={model}")

    def log_llm_response(self, raw_response: str, parse_success: bool, actual_response_len: Optional[int] = None):
        self.logger.info(f"LLM_RESPONSE #{self._eval_count} parse_success={parse_success}")

    def log_semantic_evaluation(self, evaluation: Any):
        self.logger.info(f"SEMANTIC_EVAL holistic={evaluation.holistic_score:.3f}")

    def log_reward_computation(self, index: int, completion_preview: str, expected_judgment: str,
                               extracted_judgment: str, correctness_score: float, semantic_score: float,
                               final_reward: float, prompt: str = "", full_completion: str = ""):
        self.logger.info(f"REWARD idx={index} expected={expected_judgment} extracted={extracted_judgment} FINAL={final_reward:.4f}")

    def log_batch_summary(self, batch_size: int, rewards: list, mean_reward: float, reward_std: float,
                          correct_count: int = 0, failure_count: int = 0):
        # Always include correct count in summary for dashboard parsing
        self.logger.info(f"BATCH_SUMMARY size={batch_size} mean_reward={mean_reward:.4f} correct={correct_count}/{batch_size}")
        if failure_count > 0:
            self.logger.warning(f"Batch: {correct_count}/{batch_size} correct, reward={mean_reward:.3f}±{reward_std:.3f}")


_eval_logger: Optional[EvaluationLogger] = None


def get_eval_logger() -> EvaluationLogger:
    """Get the global evaluation logger."""
    global _eval_logger
    if _eval_logger is None:
        _eval_logger = EvaluationLogger()
    return _eval_logger
