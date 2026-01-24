"""
Structured Run Logger - Thread-safe, async-compatible logging for training runs.

Provides:
- One directory per run with all related logs
- Structured JSONL metrics for easy analysis
- Config snapshots for reproducibility
- Run linking for resume chains
- Thread-safe writing (compatible with async evaluator patterns)

Usage:
    logger = RunLogger(
        experiment_name="judgment_v2",
        config={"batch_size": 2, "num_generations": 8, ...},
        resume_from="logs/runs/20260123_170000_judgment_v2"
    )
    
    # Log training events
    logger.training.info("Step 51: Starting generation")
    
    # Log Gemini calls (from async threads)
    logger.gemini.info(f"[step=51] PARALLEL REQUEST FIRED: 8 calls")
    
    # Log structured metrics (thread-safe)
    logger.log_step_metrics(StepMetrics(step=51, reward_mean=0.65, ...))
    
    # On completion
    logger.finalize(status="completed", final_step=1000)
"""

import json
import logging
import os
import threading
import queue
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from logging.handlers import QueueHandler, QueueListener


@dataclass
class StepMetrics:
    """Structured metrics for a single training step."""
    step: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Rewards
    reward_mean: Optional[float] = None
    reward_std: Optional[float] = None
    reward_min: Optional[float] = None
    reward_max: Optional[float] = None
    rewards: Optional[List[float]] = None
    
    # Correctness
    correct_count: Optional[int] = None
    total_count: Optional[int] = None
    
    # Training
    loss: Optional[float] = None
    grad_norm: Optional[float] = None
    learning_rate: Optional[float] = None
    
    # Timing
    gen_time_s: Optional[float] = None
    eval_time_s: Optional[float] = None
    update_time_s: Optional[float] = None
    total_time_s: Optional[float] = None
    
    # Diagnostics
    seq_log_prob_mean: Optional[float] = None
    seq_log_prob_std: Optional[float] = None
    token_count_mean: Optional[float] = None
    token_count_std: Optional[float] = None
    advantages: Optional[List[float]] = None
    
    # Evaluation details
    eval_method: Optional[str] = None  # "parallel" or "batch"
    eval_success: Optional[bool] = None
    
    # Inference tracking
    local_inference_ids: Optional[List[str]] = None   # e.g., ["aining.51.L.0", "aining.51.L.1", ...]
    remote_inference_ids: Optional[List[str]] = None  # e.g., ["aining.51.R.0", "aining.51.R.1", ...]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class InferenceRecord:
    """
    Record of a single inference operation (local or remote).
    
    ID Hierarchy:
    - run_id: Unique training run (e.g., "20260123_221532_training")
    - continuation_id: Previous checkpoint if resuming (e.g., "models/.../checkpoint-50")
    - step_id: Training step number (e.g., 51)
    - inference_id: Unique inference ID (e.g., "aining.51.L.3" or "aining.51.R.2")
    
    inference_id format: {run_suffix}.{step}.{type}.{idx}
    - run_suffix: Last 6 chars of run_id for readability
    - step: Step number
    - type: "L" (Local/model gen) or "R" (Remote/Gemini eval)
    - idx: Index within the batch (0-based)
    """
    # IDs
    run_id: str
    step_id: int
    inference_id: str
    inference_type: str  # "local" or "remote"
    inference_idx: int   # Index within batch
    
    # Timing
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_s: Optional[float] = None
    
    # Status
    status: str = "pending"  # "pending", "running", "completed", "failed"
    error: Optional[str] = None
    
    # Local inference fields
    token_count: Optional[int] = None
    char_count: Optional[int] = None
    
    # Remote inference fields
    finish_reason: Optional[str] = None
    response_len: Optional[int] = None
    response_id: Optional[str] = None
    
    # Evaluation results (for remote)
    judgment_correct: Optional[bool] = None
    scores: Optional[Dict[str, float]] = None
    is_fallback: Optional[bool] = None
    
    # Content reference (path to full text, if saved)
    content_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass  
class RunConfig:
    """Training run configuration snapshot."""
    # Data
    data_path: str = ""
    output_dir: str = ""
    
    # Training params
    num_steps: int = 0
    batch_size: int = 0
    num_generations: int = 0
    learning_rate: float = 0.0
    warmup_ratio: float = 0.0
    save_steps: int = 0
    
    # Model
    model_name: str = ""
    
    # Resume
    resume_from_checkpoint: Optional[str] = None
    
    # Extra
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunSummary:
    """Summary written on run completion."""
    run_id: str
    experiment_name: str
    status: str  # "completed", "interrupted", "failed"
    
    start_time: str
    end_time: str
    duration_seconds: float
    
    start_step: int
    final_step: int
    total_steps_trained: int
    
    final_reward_mean: Optional[float] = None
    final_loss: Optional[float] = None
    
    checkpoints_saved: List[str] = field(default_factory=list)
    parent_run: Optional[str] = None
    
    error_message: Optional[str] = None


class ThreadSafeJSONLWriter:
    """Thread-safe JSONL file writer using a queue."""
    
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()
    
    def _writer_loop(self):
        """Background thread that writes queued items to file."""
        with open(self.filepath, 'a') as f:
            while not self._stop_event.is_set():
                try:
                    item = self._queue.get(timeout=0.1)
                    if item is None:  # Poison pill
                        break
                    f.write(json.dumps(item) + '\n')
                    f.flush()
                except queue.Empty:
                    continue
    
    def write(self, data: Dict[str, Any]):
        """Queue data for writing (thread-safe, non-blocking)."""
        self._queue.put(data)
    
    def close(self):
        """Stop the writer thread and flush remaining items."""
        self._queue.put(None)  # Poison pill
        self._stop_event.set()
        self._writer_thread.join(timeout=5.0)


class RunLogger:
    """
    Manages structured logging for a training run.
    
    Thread-safe and async-compatible - safe to use from multiple threads
    including async evaluator background threads.
    """
    
    LOGS_ROOT = Path("logs/runs")
    
    def __init__(
        self,
        experiment_name: str,
        config: Optional[RunConfig] = None,
        resume_from: Optional[str] = None,
        log_level: str = "INFO",
    ):
        """
        Initialize a new training run logger.
        
        Args:
            experiment_name: Short name for the experiment (e.g., "judgment_v2")
            config: Training configuration to snapshot
            resume_from: Path to parent run directory if resuming
            log_level: Logging level for file handlers
        """
        self.experiment_name = experiment_name
        self.start_time = datetime.now()
        self.run_id = f"{self.start_time:%Y%m%d_%H%M%S}_{experiment_name}"
        self.run_dir = self.LOGS_ROOT / self.run_id
        self.resume_from = resume_from
        self.log_level = getattr(logging, log_level.upper())
        
        # Track state
        self._start_step = 0
        self._current_step = 0
        self._checkpoints: List[str] = []
        self._inherited_checkpoints: List[str] = []  # From parent run
        self._lock = threading.Lock()
        self._finalized = False  # Prevent double-finalization on crash
        
        # Create directory structure
        self._setup_directories()
        
        # Save config
        if config:
            self._save_config(config)
            if config.resume_from_checkpoint:
                self._start_step = self._extract_checkpoint_step(config.resume_from_checkpoint)
        
        # Link to parent run if resuming
        if resume_from:
            self._link_parent(resume_from)
        
        # Setup loggers with queue handlers (async-safe)
        self._setup_loggers()
        
        # Setup metrics writer (thread-safe)
        self._metrics_writer = ThreadSafeJSONLWriter(self.run_dir / "metrics.jsonl")
        
        # Setup inference writer (thread-safe) - for individual inference tracking
        self._inference_writer = ThreadSafeJSONLWriter(self.run_dir / "inferences.jsonl")
        
        # Update symlink and index
        self._update_current_symlink()
        self._update_index()
        
        # Log initialization
        self.training.info(f"{'='*70}")
        self.training.info(f"RUN INITIALIZED: {self.run_id}")
        self.training.info(f"{'='*70}")
        self.training.info(f"Run directory: {self.run_dir}")
        if resume_from:
            self.training.info(f"Resuming from: {resume_from}")
        self.training.info(f"Start step: {self._start_step}")
    
    def _setup_directories(self):
        """Create the run directory structure."""
        self.LOGS_ROOT.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "content").mkdir(parents=True, exist_ok=True)
    
    def _save_config(self, config: RunConfig):
        """Save configuration snapshot."""
        config_path = self.run_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump({
                "run_id": self.run_id,
                "experiment_name": self.experiment_name,
                "start_time": self.start_time.isoformat(),
                "config": config.to_dict(),
            }, f, indent=2)
    
    def _link_parent(self, parent_path: str):
        """Create link to parent run and inherit checkpoint history."""
        parent_link = self.run_dir / "parent_run.txt"
        with open(parent_link, 'w') as f:
            f.write(parent_path)
        
        # Inherit checkpoints from parent run chain
        self._inherited_checkpoints = self._load_checkpoint_chain(parent_path)
    
    def _load_checkpoint_chain(self, resume_path: str) -> List[str]:
        """
        Discover all checkpoints from the resume path.
        
        Handles two cases:
        1. resume_path is a checkpoint dir (e.g., models/foo/checkpoint-50)
           -> scan parent dir for all checkpoint-* siblings
        2. resume_path is a run log dir 
           -> load from checkpoints.json/summary.json
        """
        import re
        checkpoints = []
        path = Path(resume_path)
        
        # Case 1: It's a model checkpoint directory
        if path.name.startswith("checkpoint-") and path.exists():
            # Scan parent directory for all checkpoint-* siblings
            parent_dir = path.parent
            for item in parent_dir.iterdir():
                if item.is_dir() and item.name.startswith("checkpoint-"):
                    checkpoints.append(str(item))
        
        # Case 2: It's a run log directory with checkpoints.json
        elif (path / "checkpoints.json").exists():
            try:
                with open(path / "checkpoints.json") as f:
                    data = json.load(f)
                    for cp in data.get("checkpoints", []):
                        if cp.get("path"):
                            checkpoints.append(cp["path"])
            except Exception:
                pass
        
        # Case 3: It's a run log directory with summary.json
        elif (path / "summary.json").exists():
            try:
                with open(path / "summary.json") as f:
                    data = json.load(f)
                    checkpoints.extend(data.get("checkpoints_saved", []))
            except Exception:
                pass
        
        # Sort by step number
        def extract_step(p):
            m = re.search(r'checkpoint-(\d+)', str(p))
            return int(m.group(1)) if m else 0
        
        checkpoints.sort(key=extract_step)
        
        # Filter to only checkpoints <= our resume point
        resume_step = extract_step(resume_path)
        checkpoints = [cp for cp in checkpoints if extract_step(cp) <= resume_step]
        
        return checkpoints
    
    def _extract_checkpoint_step(self, checkpoint_path: str) -> int:
        """Extract step number from checkpoint path and return NEXT step to run."""
        import re
        match = re.search(r'checkpoint-?(\d+)', checkpoint_path)
        # Checkpoint-N means step N was completed, so we resume at N+1
        return (int(match.group(1)) + 1) if match else 0
    
    def _setup_loggers(self):
        """
        Setup loggers with QueueHandlers for async safety.
        
        Each logger writes to its own file via a queue, ensuring
        thread-safe operation from async contexts.
        """
        self._log_queues = {}
        self._queue_listeners = {}
        
        # Only training log is run-specific
        # gemini and evaluation logs are session-level (in logs/*.log)
        for name in ["training"]:
            # Create logger
            logger = logging.getLogger(f"run.{self.run_id}.{name}")
            logger.setLevel(self.log_level)
            logger.propagate = False  # Don't propagate to root
            
            # Clear any existing handlers
            logger.handlers.clear()
            
            # Create queue and handler
            log_queue = queue.Queue()
            queue_handler = QueueHandler(log_queue)
            logger.addHandler(queue_handler)
            
            # Create file handler
            file_handler = logging.FileHandler(self.run_dir / f"{name}.log")
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            
            # Create and start listener
            listener = QueueListener(log_queue, file_handler, respect_handler_level=True)
            listener.start()
            
            # Store references
            self._log_queues[name] = log_queue
            self._queue_listeners[name] = listener
            setattr(self, name, logger)
    
    def _update_current_symlink(self):
        """Update the 'current' symlink to point to this run."""
        current_link = self.LOGS_ROOT.parent / "current"
        
        # Remove existing symlink
        if current_link.is_symlink():
            current_link.unlink()
        elif current_link.exists():
            # It's a file/dir, not symlink - remove it
            import shutil
            if current_link.is_dir():
                shutil.rmtree(current_link)
            else:
                current_link.unlink()
        
        # Create new symlink (relative path for portability)
        current_link.symlink_to(f"runs/{self.run_id}")
    
    def _update_index(self):
        """Update the runs index file."""
        index_path = self.LOGS_ROOT.parent / "index.json"
        
        # Load existing index
        if index_path.exists():
            with open(index_path, 'r') as f:
                index = json.load(f)
        else:
            index = {"runs": []}
        
        # Add this run
        index["runs"].append({
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "start_time": self.start_time.isoformat(),
            "run_dir": str(self.run_dir),
            "resume_from": self.resume_from,
        })
        
        # Save index
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
    
    def log_step_metrics(self, metrics: StepMetrics):
        """
        Log structured metrics for a step (thread-safe).
        
        Can be called from any thread including async evaluator.
        """
        with self._lock:
            self._current_step = max(self._current_step, metrics.step)
        
        self._metrics_writer.write(metrics.to_dict())
        
        # Also log summary to training log
        parts = [f"Step {metrics.step}:"]
        if metrics.reward_mean is not None:
            parts.append(f"reward={metrics.reward_mean:.3f}")
        if metrics.correct_count is not None and metrics.total_count is not None:
            parts.append(f"correct={metrics.correct_count}/{metrics.total_count}")
        if metrics.loss is not None:
            parts.append(f"loss={metrics.loss:+.4f}")
        if metrics.grad_norm is not None:
            parts.append(f"grad={metrics.grad_norm:.2f}")
        if metrics.total_time_s is not None:
            parts.append(f"time={metrics.total_time_s:.1f}s")
        
        self.training.info(" | ".join(parts))
    
    def log_inference(self, record: 'InferenceRecord') -> None:
        """
        Log a single inference operation (thread-safe).
        
        Writes to inferences.jsonl for detailed tracking of each
        local (model generation) and remote (Gemini evaluation) inference.
        
        Args:
            record: InferenceRecord with inference details
        """
        self._inference_writer.write(record.to_dict())
    
    def save_content(
        self,
        inference_id: str,
        content_type: str,
        content: str,
    ) -> str:
        """
        Save full content (prompt/completion/response) to content directory.
        
        Args:
            inference_id: Unique inference ID (e.g., "aining.51.L.0")
            content_type: Type of content ("prompt", "completion", "response", "full")
            content: The full text content to save
            
        Returns:
            Relative path to the saved content file
        """
        # Sanitize inference_id for filename
        safe_id = inference_id.replace("/", "_").replace("..", "_")
        filename = f"{safe_id}_{content_type}.txt"
        filepath = self.run_dir / "content" / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"content/{filename}"
    
    def log_checkpoint(self, checkpoint_path: str, step: int):
        """Record a saved checkpoint."""
        with self._lock:
            self._checkpoints.append(checkpoint_path)
        
        # Update checkpoints manifest
        manifest_path = self.run_dir / "checkpoints.json"
        with self._lock:
            manifest = {
                "checkpoints": [
                    {"path": cp, "step": self._extract_checkpoint_step(cp)}
                    for cp in self._checkpoints
                ]
            }
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        self.training.info(f"Checkpoint saved: {checkpoint_path}")
    
    def finalize(
        self,
        status: str = "completed",
        final_step: Optional[int] = None,
        final_reward_mean: Optional[float] = None,
        final_loss: Optional[float] = None,
        error_message: Optional[str] = None,
    ):
        """
        Finalize the run and write summary.
        
        Call this when training completes (or is interrupted/fails).
        Idempotent - safe to call multiple times.
        """
        # Guard against double-finalization (from atexit + normal path)
        if self._finalized:
            return
        self._finalized = True
        
        end_time = datetime.now()
        
        with self._lock:
            step = final_step or self._current_step
            checkpoints = list(self._checkpoints)
        
        # Combine inherited and new checkpoints
        all_checkpoints = self._inherited_checkpoints + checkpoints
        
        summary = RunSummary(
            run_id=self.run_id,
            experiment_name=self.experiment_name,
            status=status,
            start_time=self.start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=(end_time - self.start_time).total_seconds(),
            start_step=self._start_step,
            final_step=step,
            total_steps_trained=step - self._start_step,
            final_reward_mean=final_reward_mean,
            final_loss=final_loss,
            checkpoints_saved=all_checkpoints,
            parent_run=self.resume_from,
            error_message=error_message,
        )
        
        # Write summary
        summary_path = self.run_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(asdict(summary), f, indent=2)
        
        self.training.info(f"{'='*70}")
        self.training.info(f"RUN FINALIZED: {status}")
        self.training.info(f"{'='*70}")
        self.training.info(f"Duration: {summary.duration_seconds:.1f}s")
        self.training.info(f"Steps trained: {summary.total_steps_trained}")
        inherited = len(self._inherited_checkpoints)
        new = len(checkpoints)
        if inherited > 0:
            self.training.info(f"Checkpoints: {inherited + new} ({inherited} inherited + {new} new)")
        else:
            self.training.info(f"Checkpoints: {new}")
        
        # Cleanup
        self._cleanup()
    
    def _cleanup(self):
        """Stop queue listeners and close writers."""
        # Stop queue listeners
        for listener in self._queue_listeners.values():
            listener.stop()
        
        # Close writers
        self._metrics_writer.close()
        self._inference_writer.close()
    
    @classmethod
    def get_run_chain(cls, run_dir: str) -> List[str]:
        """
        Get the full chain of runs leading to this one.
        
        Follows parent_run.txt links back to the original run.
        Returns list from oldest to newest.
        """
        chain = [run_dir]
        current = Path(run_dir)
        
        while True:
            parent_link = current / "parent_run.txt"
            if not parent_link.exists():
                break
            
            parent_path = parent_link.read_text().strip()
            chain.insert(0, parent_path)
            current = Path(parent_path)
        
        return chain
    
    @classmethod
    def load_metrics(cls, run_dir: str) -> List[Dict[str, Any]]:
        """Load all metrics from a run's metrics.jsonl file."""
        metrics_path = Path(run_dir) / "metrics.jsonl"
        if not metrics_path.exists():
            return []
        
        metrics = []
        with open(metrics_path, 'r') as f:
            for line in f:
                if line.strip():
                    metrics.append(json.loads(line))
        return metrics
    
    @classmethod
    def load_full_history(cls, run_dir: str) -> List[Dict[str, Any]]:
        """Load metrics from entire run chain (for resumed runs)."""
        chain = cls.get_run_chain(run_dir)
        all_metrics = []
        
        for rd in chain:
            all_metrics.extend(cls.load_metrics(rd))
        
        # Sort by step
        all_metrics.sort(key=lambda m: m.get("step", 0))
        return all_metrics
