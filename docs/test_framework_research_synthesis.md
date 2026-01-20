# Test Framework Research Synthesis

*Research conducted on cognitive-model-integration-and-training-dataset-design-v24-test-cases.csv*

## Executive Summary

The external test framework employs sophisticated techniques for training cognitive operation recognition. This document synthesizes findings for application to the Cognitive Organism project.

---

## I. Key Techniques Extracted

### 1. NX Token System (Semantic Anchoring)

**Mechanism**: 100 unique tokens with phase-encoding prefixes
- `⟢NX-01-XXXX⟣` = P1 (Attention) - 25 tokens
- `⟢NX-02-XXXX⟣` = P2 (Understanding) - 25 tokens
- `⟢NX-03-XXXX⟣` = P3 (Judgment) - 25 tokens
- `⟢NX-04-XXXX⟣` = P4 (Decision) - 25 tokens

**Function**:
- Prevents semantic drift via opaque tokens
- Phase-prefix encoding enables automatic validation
- Rare Unicode delimiters create distinct attention patterns
- **Training scaffold**: Can potentially be distilled away after learning

### 2. Distractor System (Contrastive Learning)

**10 Distractor Types**:
1. `plausible_misreading` - surface-correct but operationally wrong
2. `unsupported_inference` - conclusion without evidence
3. `overextension` - operation applied beyond scope
4. `missing_context` - ignores crucial information
5. `misaligned_phase` - **CRITICAL**: confusing P2 with P3
6. `heuristic_bias` - pattern-matching without understanding
7. `affective_leap` - emotion substituted for cognition
8. `self_reference_error` - failure in self-appropriation
9. `temporal_confusion` - aspectual misreading
10. `other` - catch-all

**Function**: Defines the negative space of correct operation

### 3. Evidence Requirement (Grounding)

**Schema requirement**: Every operation must cite textual evidence
- Exact quotation or precise paraphrase
- Prevents hallucinated justification
- Mirrors Lonergan's "reversion to data"

### 4. Dual-Packet Architecture

**Student packet**: prompt, guidance, reference_operations, distractors
**Evaluator packet**: prompt, instructions, rubric (must_include, unacceptable, partial_credit)

**Function**: Role separation enables two-model training

### 5. Rubric Structure (Virtually Unconditioned Mapping)

```
must_include    → conditions that must be fulfilled
unacceptable    → conditions whose presence negates judgment
partial_credit  → partial fulfillment states
```

**Function**: Maps directly to Lonergan's judgment structure

### 6. Multi-Dimensional Coverage

- **20 domains** (epistemology through aesthetic)
- **4 voices** (first/second/third/dialogical)
- **5 difficulties** (introductory through expert)
- **10 temporal frames** (present_continuous, aorist, iterative, perfect, inchoative, potential, compound, retrospective, prospective, atemporal)

**Function**: Forces systematic variation, prevents clustering

### 7. P1 Adaptation ("Descriptive Restatement")

**Problem**: LLMs have no sensory experience
**Solution**: P1 = "disciplined descriptive restatement of linguistic input"

**Function**: Adapts experiential P1 to linguistic domain while preserving operational structure

### 8. Self-Similar Design

The framework uses itself on itself:
- Source documents are P1 presentations
- Explication provides P2 insight
- Schema validation provides P3 criteria
- Generator must P4 decide to follow norms

---

## II. Comparison with Current Project

| Dimension | External Framework | Cognitive Organism |
|-----------|-------------------|-------------------|
| **Task** | Recognize operations in text | Perform operations on input |
| **Output** | Structured analysis | Cognitive output (judgment) |
| **Anchoring** | NX tokens (100) | None currently |
| **Distractors** | 10 typed categories | None currently |
| **Evidence** | Required textual grounding | Condition fulfillment reasoning |
| **Difficulty scaling** | 5 levels explicit | Implicit in domain variety |
| **Temporal frames** | 10 aspectual categories | None currently |
| **Rubric structure** | must/unacceptable/partial | Binary (correct/incorrect) |

---

## III. Gaps in Current Project

1. **No semantic anchoring tokens** - operations identified only by natural language
2. **No distractor system** - only positive examples in training
3. **No difficulty scaling** - all examples treated equivalently
4. **No temporal/aspectual dimension** - ignores how operations manifest in different temporal frames
5. **Binary reward structure** - no partial credit mechanism
6. **No coverage enforcement** - domain/voice/difficulty distribution uncontrolled

---

## IV. Materials for Insight Generation

### From Prompt Engineering Analysis:
- Dual-encoding (token + label) creates redundancy that aids learning
- Phase-token alignment enables automatic validation
- Guidance field separates meta-instruction from content
- Coverage planning is explicit instruction, not implicit hope

### From Semantic Anchoring Analysis:
- Opaque tokens prevent lexical interference from pre-training
- Enumeration constraints create "semantic type safety"
- Evidence requirement operationalizes "grounding in data"
- Distractor types operationalize cognitive error categories

### From Cognitive Model Integration Analysis:
- Source documents arranged in genetic sequence (whole before parts)
- "Descriptive restatement" solves the LLM-experience problem
- Schema structure embodies virtually unconditioned
- Self-similar design means framework models what it trains

### Training Scaffold Concept:
- NX tokens are maximally distinct during training
- After learning, operational patterns may be distillable
- Scaffolds optimize learning, not necessarily inference
- Similar to curriculum learning / auxiliary tasks

### Contrastive Learning Concept:
- Distractors define operation boundaries via negation
- `misaligned_phase` specifically targets P2/P3 confusion
- Learning what X is NOT helps learn what X IS
- Negative examples as important as positive examples
