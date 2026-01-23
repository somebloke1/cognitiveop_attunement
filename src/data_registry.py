"""
Data Registry for Generated Training Samples

Tracks all generated samples with metadata:
- When generated
- Which model generated them
- Domain/difficulty/judgment distribution
- File locations

Append-only log ensures no data loss and full provenance.
"""

import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional
import hashlib


@dataclass
class GenerationRun:
    """Record of a single generation run."""

    run_id: str
    timestamp: str
    model: str
    num_examples: int
    output_file: str
    domains: List[str]
    difficulties: List[str]
    judgments: List[str]
    config: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @classmethod
    def create(
        cls,
        model: str,
        num_examples: int,
        output_file: str,
        domains: List[str],
        difficulties: List[str],
        judgments: List[str],
        config: Optional[Dict[str, Any]] = None,
        notes: str = "",
    ) -> "GenerationRun":
        """Create a new generation run record."""
        timestamp = datetime.now().isoformat()
        # Create unique run ID from timestamp + model
        run_id = hashlib.sha256(
            f"{timestamp}:{model}:{output_file}".encode()
        ).hexdigest()[:12]

        return cls(
            run_id=run_id,
            timestamp=timestamp,
            model=model,
            num_examples=num_examples,
            output_file=output_file,
            domains=domains,
            difficulties=difficulties,
            judgments=judgments,
            config=config or {},
            notes=notes,
        )


class DataRegistry:
    """
    Append-only registry of all data generation runs.

    Provides:
    - Full provenance tracking
    - No overwrites (each run gets unique ID and timestamped file)
    - Statistics on what's been generated
    - Easy lookup of samples by model/domain/etc.
    """

    def __init__(self, registry_path: Optional[Path] = None):
        self.registry_path = registry_path or Path("data/registry.jsonl")
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

    def log_run(self, run: GenerationRun) -> None:
        """Append a generation run to the registry."""
        with open(self.registry_path, "a") as f:
            f.write(json.dumps(asdict(run)) + "\n")

    def get_all_runs(self) -> List[GenerationRun]:
        """Load all generation runs from the registry."""
        runs = []
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        runs.append(GenerationRun(**data))
        return runs

    def get_runs_by_model(self, model: str) -> List[GenerationRun]:
        """Get all runs for a specific model."""
        return [r for r in self.get_all_runs() if model in r.model]

    def get_total_examples(self) -> int:
        """Get total number of examples generated across all runs."""
        return sum(r.num_examples for r in self.get_all_runs())

    def get_examples_by_model(self) -> Dict[str, int]:
        """Get count of examples per model."""
        counts: Dict[str, int] = {}
        for run in self.get_all_runs():
            model = run.model
            counts[model] = counts.get(model, 0) + run.num_examples
        return counts

    def get_summary(self) -> str:
        """Get a human-readable summary of the registry."""
        runs = self.get_all_runs()
        if not runs:
            return "No generation runs recorded."

        lines = [
            "=" * 60,
            "DATA REGISTRY SUMMARY",
            "=" * 60,
            f"Total runs: {len(runs)}",
            f"Total examples: {self.get_total_examples()}",
            "",
            "By model:",
        ]

        for model, count in sorted(self.get_examples_by_model().items()):
            lines.append(f"  {model}: {count} examples")

        lines.extend(
            [
                "",
                "Recent runs:",
            ]
        )

        for run in runs[-5:]:
            lines.append(
                f"  [{run.run_id}] {run.timestamp[:16]} | {run.model} | "
                f"{run.num_examples} examples -> {run.output_file}"
            )

        return "\n".join(lines)

    def generate_unique_filename(self, prefix: str, model: str) -> str:
        """Generate a unique filename for a new generation run."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize model name for filename
        model_short = model.replace("/", "_").replace("-", "_").split("_")[0]
        return f"{prefix}_{model_short}_{timestamp}.jsonl"


def get_registry() -> DataRegistry:
    """Get the default data registry."""
    return DataRegistry(Path("data/registry.jsonl"))
