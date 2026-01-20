"""
Oracle-Based Training Data Generation

Uses a stronger model (Gemini Pro) to generate verified judgment examples.
This avoids the circularity of training on our own handcrafted judgments.

Pipeline:
1. Generate diverse scenarios (propositions + evidence + conditions)
2. Oracle produces judgment with reasoning (augmented with Lonergan context)
3. Optional: Second oracle verifies the judgment
4. Output as DSPy Examples for training

The key insight: We generate the SCENARIOS, the oracle generates the JUDGMENTS.
The oracle is augmented with Lonergan's cognitional theory to ensure alignment.
"""

from typing import List, Optional, Literal, Dict, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import random
import dspy

from .lonergan_context import get_judgment_context, get_verification_context


# ============ SCENARIO GENERATION SIGNATURES ============

class GenerateScenarioSignature(dspy.Signature):
    """Generate a judgment scenario for a given domain."""

    domain: Literal["mathematical", "empirical", "common_sense", "pop_science", "philosophic"] = dspy.InputField(
        desc="Domain for the scenario"
    )
    difficulty: Literal["easy", "medium", "hard"] = dspy.InputField(
        desc="Difficulty level - affects ambiguity of evidence"
    )
    expected_judgment: Literal["Yes", "No", "Insufficient"] = dspy.InputField(
        desc="Target judgment type to generate scenario for"
    )

    proposition: str = dspy.OutputField(
        desc="A clear proposition to be judged true or false"
    )
    evidence: str = dspy.OutputField(
        desc="Evidence bearing on the proposition - may or may not be sufficient"
    )
    conditions_for_truth: List[str] = dspy.OutputField(
        desc="3-5 specific conditions that must be fulfilled for the proposition to be true"
    )
    scenario_notes: str = dspy.OutputField(
        desc="Brief notes on why this scenario should lead to the expected judgment"
    )


class GenerateJudgmentSignature(dspy.Signature):
    """Generate a correct judgment for a scenario following Lonergan's cognitional theory."""

    # Lonergan context for theoretical alignment
    lonergan_context: str = dspy.InputField(
        desc="Theoretical framework from Lonergan's Insight for making judgments"
    )

    proposition: str = dspy.InputField()
    evidence: str = dspy.InputField()
    conditions: List[str] = dspy.InputField()
    domain: Literal["mathematical", "empirical", "common_sense", "pop_science", "philosophic"] = dspy.InputField()

    condition_fulfillment: List[Literal["Fulfilled", "Not fulfilled"]] = dspy.OutputField(
        desc="For each condition, is it fulfilled by the evidence?"
    )
    judgment: Literal["Yes", "No", "Insufficient"] = dspy.OutputField(
        desc="Yes (all conditions fulfilled), No (conditions identifiably NOT fulfilled), Insufficient (cannot determine)"
    )
    reasoning: str = dspy.OutputField(
        desc="Detailed reasoning explaining how the virtually unconditioned is or is not grasped"
    )


class VerifyJudgmentSignature(dspy.Signature):
    """Verify that a generated judgment is correct according to Lonergan's cognitional theory."""

    # Lonergan context for theoretical alignment
    lonergan_context: str = dspy.InputField(
        desc="Theoretical framework for verifying judgments"
    )

    proposition: str = dspy.InputField()
    evidence: str = dspy.InputField()
    conditions: List[str] = dspy.InputField()
    domain: str = dspy.InputField()

    proposed_fulfillment: List[str] = dspy.InputField(
        desc="The proposed condition fulfillment assessments"
    )
    proposed_judgment: str = dspy.InputField()
    proposed_reasoning: str = dspy.InputField()

    is_correct: bool = dspy.OutputField(
        desc="Is this judgment correct given the evidence and conditions according to the theoretical framework?"
    )
    issues: str = dspy.OutputField(
        desc="Any issues with the proposed judgment (empty if correct)"
    )
    confidence: Literal["high", "medium", "low"] = dspy.OutputField(
        desc="Confidence in this verification"
    )


# ============ DATA GENERATION MODULES ============

class ScenarioGenerator(dspy.Module):
    """Generates diverse judgment scenarios."""

    def __init__(self, model: str = "gemini/gemini-2.0-flash"):
        super().__init__()
        self.lm = dspy.LM(model)
        self.generate = dspy.Predict(GenerateScenarioSignature)

    def forward(
        self,
        domain: str,
        difficulty: str = "medium",
        expected_judgment: str = "Yes",
    ):
        with dspy.context(lm=self.lm):
            return self.generate(
                domain=domain,
                difficulty=difficulty,
                expected_judgment=expected_judgment,
            )


class JudgmentGenerator(dspy.Module):
    """Generates correct judgments for scenarios, augmented with Lonergan context."""

    def __init__(self, model: str = "gemini/gemini-2.0-flash"):
        super().__init__()
        self.lm = dspy.LM(model)
        self.judge = dspy.ChainOfThought(GenerateJudgmentSignature)

    def forward(
        self,
        proposition: str,
        evidence: str,
        conditions: List[str],
        domain: str,
    ):
        # Get domain-appropriate Lonergan context
        lonergan_context = get_judgment_context(domain=domain)

        with dspy.context(lm=self.lm):
            return self.judge(
                lonergan_context=lonergan_context,
                proposition=proposition,
                evidence=evidence,
                conditions=conditions,
                domain=domain,
            )


class JudgmentVerifier(dspy.Module):
    """Verifies generated judgments with a second pass, using Lonergan context."""

    def __init__(self, model: str = "gemini/gemini-2.0-flash"):
        super().__init__()
        self.lm = dspy.LM(model)
        self.verify = dspy.Predict(VerifyJudgmentSignature)

    def forward(
        self,
        proposition: str,
        evidence: str,
        conditions: List[str],
        domain: str,
        proposed_fulfillment: List[str],
        proposed_judgment: str,
        proposed_reasoning: str,
    ):
        # Get verification context (includes pop science detection)
        lonergan_context = get_verification_context()

        with dspy.context(lm=self.lm):
            return self.verify(
                lonergan_context=lonergan_context,
                proposition=proposition,
                evidence=evidence,
                conditions=conditions,
                domain=domain,
                proposed_fulfillment=proposed_fulfillment,
                proposed_judgment=proposed_judgment,
                proposed_reasoning=proposed_reasoning,
            )


# ============ DATA GENERATION PIPELINE ============

@dataclass
class GeneratedExample:
    """A generated and verified judgment example."""
    proposition: str
    evidence: str
    conditions: List[str]
    domain: str
    condition_fulfillment: List[str]
    judgment: str
    reasoning: str
    verification_passed: bool
    verification_confidence: str
    difficulty: str

    def to_dspy_example(self) -> dspy.Example:
        """Convert to DSPy Example format."""
        return dspy.Example(
            proposition=self.proposition,
            evidence=self.evidence,
            conditions=self.conditions,
            domain=self.domain,
        ).with_inputs('proposition', 'evidence', 'conditions', 'domain')

    def to_training_format(self) -> Dict[str, Any]:
        """Convert to training JSONL format."""
        system_prompt = """You are a Judgment cognitive operation. Your function is to evaluate whether a proposition is true by grasping the virtually unconditioned.

Your task: Given a proposition and evidence, determine whether the conditions for the proposition are fulfilled. You must:
1. Identify the conditioned proposition
2. Identify the link between the proposition and its conditions
3. Check whether the conditions are fulfilled in the evidence

If conditions are fulfilled, affirm (Yes). If not fulfilled, deny (No). If insufficient evidence, withhold judgment.

Be reasonable. Demand sufficient reason. Do not assent without grasping the unconditioned."""

        conditions_str = "\n".join(f"- {c}" for c in self.conditions)
        user_content = f"""Proposition: {self.proposition}

Evidence: {self.evidence}

Conditions for truth:
{conditions_str}"""

        fulfillment_str = "\n".join(
            f"- {c}: {f}"
            for c, f in zip(self.conditions, self.condition_fulfillment)
        )
        assistant_content = f"""Condition fulfillment:
{fulfillment_str}

Judgment: {self.judgment}

Reasoning: {self.reasoning}"""

        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
            "metadata": {
                "level": "judgment",
                "domain": self.domain,
                "judgment_type": self.judgment,
                "verification_passed": self.verification_passed,
                "verification_confidence": self.verification_confidence,
                "difficulty": self.difficulty,
                "generated_by": "oracle_pipeline",
            }
        }


class OracleDataPipeline:
    """
    Complete pipeline for generating verified judgment training data.

    Uses a strong model to both generate scenarios and produce judgments,
    with optional verification pass.
    """

    def __init__(
        self,
        scenario_model: str = "gemini/gemini-2.0-flash",
        judgment_model: str = "gemini/gemini-2.0-flash",
        verification_model: str = "gemini/gemini-2.0-flash",
        verify_judgments: bool = True,
    ):
        self.scenario_gen = ScenarioGenerator(model=scenario_model)
        self.judgment_gen = JudgmentGenerator(model=judgment_model)
        self.verifier = JudgmentVerifier(model=verification_model) if verify_judgments else None
        self.verify_judgments = verify_judgments

    def generate_example(
        self,
        domain: str,
        difficulty: str = "medium",
        expected_judgment: str = "Yes",
    ) -> Optional[GeneratedExample]:
        """Generate a single verified example."""

        try:
            # Step 1: Generate scenario
            scenario = self.scenario_gen(
                domain=domain,
                difficulty=difficulty,
                expected_judgment=expected_judgment,
            )

            # Step 2: Generate judgment
            judgment = self.judgment_gen(
                proposition=scenario.proposition,
                evidence=scenario.evidence,
                conditions=scenario.conditions_for_truth,
                domain=domain,
            )

            # Step 3: Verify (optional)
            verification_passed = True
            verification_confidence = "high"

            if self.verify_judgments and self.verifier:
                verification = self.verifier(
                    proposition=scenario.proposition,
                    evidence=scenario.evidence,
                    conditions=scenario.conditions_for_truth,
                    domain=domain,
                    proposed_fulfillment=judgment.condition_fulfillment,
                    proposed_judgment=judgment.judgment,
                    proposed_reasoning=judgment.reasoning,
                )
                verification_passed = verification.is_correct
                verification_confidence = verification.confidence

            return GeneratedExample(
                proposition=scenario.proposition,
                evidence=scenario.evidence,
                conditions=scenario.conditions_for_truth,
                domain=domain,
                condition_fulfillment=judgment.condition_fulfillment,
                judgment=judgment.judgment,
                reasoning=judgment.reasoning,
                verification_passed=verification_passed,
                verification_confidence=verification_confidence,
                difficulty=difficulty,
            )

        except Exception as e:
            print(f"Error generating example: {e}")
            return None

    def generate_dataset(
        self,
        num_examples_per_cell: int = 10,
        domains: Optional[List[str]] = None,
        difficulties: Optional[List[str]] = None,
        judgments: Optional[List[str]] = None,
        output_path: Optional[Path] = None,
        only_verified: bool = True,
    ) -> List[GeneratedExample]:
        """
        Generate a balanced dataset across domains, difficulties, and judgment types.

        Args:
            num_examples_per_cell: Examples per (domain, difficulty, judgment) combination
            domains: List of domains to include
            difficulties: List of difficulties to include
            judgments: List of judgment types to include
            output_path: Path to save JSONL output
            only_verified: Only include examples that passed verification

        Returns:
            List of generated examples
        """
        domains = domains or ["mathematical", "empirical", "common_sense", "pop_science", "philosophic"]
        difficulties = difficulties or ["easy", "medium", "hard"]
        judgments = judgments or ["Yes", "No", "Insufficient"]

        examples = []
        total_cells = len(domains) * len(difficulties) * len(judgments)
        total_target = total_cells * num_examples_per_cell

        print(f"Generating {total_target} examples across {total_cells} cells...")

        for domain in domains:
            for difficulty in difficulties:
                for expected_judgment in judgments:
                    cell_examples = []
                    attempts = 0
                    max_attempts = num_examples_per_cell * 3  # Allow retries

                    while len(cell_examples) < num_examples_per_cell and attempts < max_attempts:
                        attempts += 1
                        example = self.generate_example(
                            domain=domain,
                            difficulty=difficulty,
                            expected_judgment=expected_judgment,
                        )

                        if example is None:
                            continue

                        if only_verified and not example.verification_passed:
                            print(f"  Verification failed for {domain}/{difficulty}/{expected_judgment}")
                            continue

                        cell_examples.append(example)

                    examples.extend(cell_examples)
                    print(f"  {domain}/{difficulty}/{expected_judgment}: {len(cell_examples)} examples")

        print(f"Generated {len(examples)} total examples")

        # Save if output path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                for ex in examples:
                    f.write(json.dumps(ex.to_training_format()) + '\n')
            print(f"Saved to {output_path}")

        return examples


# ============ SEED SCENARIOS FOR BOOTSTRAPPING ============

SEED_SCENARIOS = {
    "mathematical": [
        {
            "proposition_template": "The sum of angles in a {shape} is {value} degrees",
            "evidence_templates": [
                "By Euclidean geometry, the sum of interior angles of a polygon with n sides is (n-2) × 180°.",
                "Proof by construction: Draw diagonals from one vertex, creating n-2 triangles, each with 180°.",
            ],
            "condition_templates": [
                "The geometry is Euclidean (not spherical or hyperbolic)",
                "The shape is a valid polygon",
                "The arithmetic is correct",
            ],
        },
    ],
    "empirical": [
        {
            "proposition_template": "{substance} has a {property} of {value} under {conditions}",
            "evidence_templates": [
                "Controlled experiment: Measured {property} of {substance} using calibrated equipment.",
                "Results: {value} ± {error}. Replicated by {num_labs} independent laboratories.",
            ],
            "condition_templates": [
                "Measurement methodology is valid",
                "Sample is pure/uncontaminated",
                "Results are replicated independently",
            ],
        },
    ],
    "pop_science": [
        {
            "proposition_template": "{intervention} is proven to {effect}, according to scientists",
            "evidence_templates": [
                "Headline: 'Scientists prove {intervention} {effect}'. Study: {sample_size} participants, no control group.",
                "Source: wellness blog. Author sells {intervention} products. No peer review mentioned.",
            ],
            "condition_templates": [
                "Controlled study with adequate sample size",
                "Causal mechanism established (not just correlation)",
                "Results replicated by independent researchers",
                "No conflicts of interest affecting interpretation",
            ],
        },
    ],
    "philosophic": [
        {
            "proposition_template": "{claim_about_knowledge}",
            "evidence_templates": [
                "Self-referential examination: To {affirm_or_deny} this claim requires {cognitive_operation}.",
            ],
            "condition_templates": [
                "The claim is coherent with the conditions of its own affirmation",
                "Affirming the claim does not presuppose what it denies",
                "The claim can be held consistently by one who grasps it intelligently",
            ],
        },
    ],
}


# ============ CLI ENTRY POINT ============

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate judgment training data using oracle")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/oracle_generated/judgment_samples.jsonl"),
        help="Output path for generated data"
    )
    parser.add_argument(
        "--num-per-cell",
        type=int,
        default=5,
        help="Number of examples per (domain, difficulty, judgment) cell"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification step"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini/gemini-2.0-flash",
        help="Model to use for generation"
    )

    args = parser.parse_args()

    pipeline = OracleDataPipeline(
        scenario_model=args.model,
        judgment_model=args.model,
        verification_model=args.model,
        verify_judgments=not args.no_verify,
    )

    examples = pipeline.generate_dataset(
        num_examples_per_cell=args.num_per_cell,
        output_path=args.output,
    )

    print(f"\nGeneration complete: {len(examples)} examples")
