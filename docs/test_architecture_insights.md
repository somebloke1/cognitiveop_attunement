# Test Architecture Insights for Cognitive Operation Training

*Generated: 2026-01-20*

This document presents powerful insights for improving the test architecture and test generation for the Cognitive Organism project. Each insight derives from the research synthesis on cognitive-model-integration training, mapped against the current project state and Lonergan's invariant cognitive pattern.

---

## Insight 1: Operational Anchoring Tokens (Training Scaffolds)

### Name
**Phase-Encoded Semantic Anchors**

### Source
External framework's NX Token System: 100 unique tokens with phase-encoding prefixes (`NX-01-XXXX` through `NX-04-XXXX`)

### Insight
Opaque, phase-encoded tokens create distinct semantic "slots" that prevent lexical interference from pre-training while enabling automatic validation. These tokens function as **training scaffolds**—maximally distinct during learning, potentially distillable after operational patterns are established.

### Mechanism
1. Define 25 unique tokens per cognitive level:
   - `[P1-ATT-01]` through `[P1-ATT-25]` for Attention
   - `[P2-INS-01]` through `[P2-INS-25]` for Understanding (Insight)
   - `[P3-JDG-01]` through `[P3-JDG-25]` for Judgment
   - `[P4-DEC-01]` through `[P4-DEC-25]` for Decision

2. Training examples require the model to produce the appropriate anchor token as part of structured output:
   ```
   Operation: [P3-JDG-17]
   Conditioned: The proposition under examination
   Link: The conditions that would warrant affirmation
   Fulfillment: Assessment of whether conditions are met
   Judgment: Yes/No/Insufficient
   ```

3. Reward function validates phase-prefix alignment (P3 token in judgment task = bonus)

4. After training stabilizes, test whether operational patterns transfer without anchor tokens (scaffold removal)

### Expected Impact
- **Prevents semantic drift**: Anchors don't carry pre-training associations
- **Enables automatic validation**: Phase prefix is machine-checkable
- **Creates attention distinctiveness**: Rare Unicode creates unique attention patterns
- **Supports curriculum**: Can gradually reduce anchor dependency

### Implementation Complexity
**Medium** - Requires vocabulary extension, modified tokenizer configuration, and reward function update. No architectural changes.

---

## Insight 2: Contrastive Distractor Learning

### Name
**Negative Space Definition Through Typed Distractors**

### Source
External framework's 10 distractor types, especially `misaligned_phase` (confusing P2 with P3)

### Insight
Learning what an operation IS requires learning what it is NOT. The current training uses only positive examples, which allows the model to learn surface patterns rather than operational boundaries. **Contrastive examples with typed failure modes define the negative space of correct operation.**

### Mechanism
1. For each positive example, generate 2-3 distractor completions with explicit failure types:

   ```json
   {
     "prompt": "...",
     "correct_completion": "...",
     "distractors": [
       {
         "type": "misaligned_phase",
         "completion": "The hypothesis seems plausible because...",
         "explanation": "Offers P2 understanding without P3 judgment"
       },
       {
         "type": "unsupported_inference",
         "completion": "Judgment: Yes. The conditions are clearly met.",
         "explanation": "Affirms without showing condition fulfillment"
       }
     ]
   }
   ```

2. Training formats:
   - **Preference learning**: Model learns to prefer correct over distractor
   - **Classification**: Model identifies which distractor type is exhibited
   - **Correction**: Model transforms distractor into correct completion

3. Reward function penalizes distractor patterns:
   - Detecting "P2 language in P3 output" triggers penalty
   - Detecting "affirmation without evidence chain" triggers penalty

### Expected Impact
- **Prevents P2/P3 conflation**: The most critical operational boundary
- **Generalizes better**: Learning from negative examples constrains hypothesis space
- **Operationalizes cognitive error categories**: Maps Lonergan's "flight from understanding" to detectable patterns

### Implementation Complexity
**High** - Requires distractor generation pipeline (could use Gemini oracle), dataset restructuring, and potentially switching to preference-based training (DPO or GRPO with preference pairs).

---

## Insight 3: Evidence Grounding Requirement

### Name
**Mandatory Textual Reversion**

### Source
External framework's schema requirement for exact quotation or precise paraphrase; Lonergan's principle that "fulfillment is found by REVERTING from formulation to the more rudimentary state"

### Insight
The current training allows the model to claim condition fulfillment without demonstrating it. **Every judgment must cite specific textual evidence from the input**—this operationalizes Lonergan's reversion to data and prevents hallucinated justification.

### Mechanism
1. Schema modification—require `evidence` field with quoted text:

   ```json
   {
     "conditioned": "Water boils at 100°C at standard pressure",
     "link": "If thermometer reads 100°C and pressure gauge reads 1 atm during boiling, then conditions met",
     "evidence": [
       "\"The thermometer displayed exactly 100.0°C\"",
       "\"Atmospheric pressure was 101.3 kPa\""
     ],
     "fulfillment": "Both conditions verified in cited evidence",
     "judgment": "Yes"
   }
   ```

2. Reward function validates evidence:
   - Check that quoted text actually appears in input (string matching)
   - Penalize fabricated quotes
   - Bonus for precise, relevant citation

3. Evidence requirements vary by domain:
   - **Mathematical**: Quote axioms, definitions, proof steps
   - **Empirical**: Quote data, measurements, observations
   - **Common sense**: Quote situational details
   - **Philosophic**: Quote the position being examined

### Expected Impact
- **Prevents hallucination**: Model cannot claim fulfillment without textual support
- **Operationalizes reversion**: Makes "reverting to data" a structural requirement
- **Enables verification**: Evidence chains are machine-checkable

### Implementation Complexity
**Medium** - Requires schema update, reward function enhancement for string matching, and regeneration of training examples with evidence fields.

---

## Insight 4: Graduated Difficulty Scaffolding

### Name
**Five-Level Difficulty Curriculum**

### Source
External framework's explicit 5 difficulty levels (introductory → expert); curriculum learning research

### Insight
The current training treats all examples equivalently, which can cause the model to overfit on easy patterns while underlearning hard cases. **Explicit difficulty tagging enables curriculum scheduling**—starting with clear cases and progressing to ambiguous ones.

### Mechanism
1. Tag each example with difficulty level (1-5):

   | Level | Characteristics | Example |
   |-------|----------------|---------|
   | 1 - Introductory | Obvious fulfillment/unfulfillment | "2+2=4" |
   | 2 - Foundational | Clear conditions, straightforward check | Standard empirical verification |
   | 3 - Intermediate | Multiple conditions, some ambiguity | Complex causal reasoning |
   | 4 - Advanced | Counterposition detection, subtle errors | Pop science claims |
   | 5 - Expert | Philosophic self-consistency, edge cases | Performative contradiction detection |

2. Curriculum schedule:
   - Epochs 1-20: Levels 1-2 only
   - Epochs 21-50: Levels 1-3
   - Epochs 51-80: Levels 1-4
   - Epochs 81+: All levels

3. Reward scaling by difficulty:
   - Correct judgment on Level 5 = 1.0x reward
   - Correct judgment on Level 1 = 0.6x reward
   - Incorrect on Level 1 = larger penalty (should be easy)

### Expected Impact
- **Prevents premature generalization**: Model masters basics before confronting edge cases
- **Improves hard-case performance**: More training signal on difficult examples
- **Enables progress tracking**: Difficulty-stratified evaluation metrics

### Implementation Complexity
**Medium** - Requires difficulty tagging (can be oracle-assisted), curriculum scheduler in training loop, and difficulty-aware reward scaling.

---

## Insight 5: Rubric-Based Partial Credit

### Name
**Virtually Unconditioned Rubric Structure**

### Source
External framework's three-part rubric (must_include, unacceptable, partial_credit); Lonergan's judgment structure (conditioned, link, fulfillment)

### Insight
Binary scoring (correct/incorrect) loses information about *how* the model fails. **A rubric that mirrors the virtually unconditioned structure enables partial credit and diagnostic evaluation.**

### Mechanism
1. Each example includes a rubric with three components:

   ```json
   {
     "rubric": {
       "must_include": [
         "Identifies the conditioned proposition",
         "States the link to conditions",
         "Assesses fulfillment with evidence",
         "Reaches appropriate judgment"
       ],
       "unacceptable": [
         "Affirms without showing fulfillment",
         "Confuses correlation with causation",
         "Appeals to authority without evidence"
       ],
       "partial_credit": {
         "correct_structure_wrong_judgment": 0.3,
         "correct_judgment_missing_evidence": 0.5,
         "identifies_conditions_but_no_judgment": 0.4
       }
     }
   }
   ```

2. Reward function applies rubric:
   - Full credit: All must_include present, no unacceptable
   - Partial credit: Based on specific partial_credit conditions
   - Zero credit: Any unacceptable present

3. Diagnostic logging:
   - Track which must_include items are most often missing
   - Track which unacceptable patterns are most common
   - Use for targeted training data generation

### Expected Impact
- **Finer gradient signal**: Model gets reward for partial correctness
- **Diagnostic capability**: Identifies specific failure modes
- **Maps to Lonergan**: Rubric structure mirrors virtually unconditioned structure

### Implementation Complexity
**High** - Requires rubric generation for all examples, sophisticated reward function with rubric evaluation, and logging infrastructure for diagnostics.

---

## Insight 6: Temporal-Aspectual Variation

### Name
**Aspectual Frame Coverage**

### Source
External framework's 10 temporal frames (present_continuous, aorist, iterative, perfect, inchoative, potential, compound, retrospective, prospective, atemporal)

### Insight
Cognitive operations manifest differently across temporal aspects. The current training ignores this dimension, which can cause the model to learn aspect-specific patterns rather than invariant operations. **Systematic variation across aspectual frames forces operational invariance.**

### Mechanism
1. Define aspectual frame markers for training data:

   | Frame | Character | Example in Judgment |
   |-------|-----------|---------------------|
   | Present continuous | Ongoing | "The experiment is showing..." |
   | Aorist | Completed point | "The result was X" |
   | Iterative | Repeated | "Across trials, consistently..." |
   | Perfect | Completed with present relevance | "The theory has been verified" |
   | Inchoative | Beginning | "Evidence is starting to suggest..." |
   | Potential | Possible | "If data confirms, judgment would be..." |
   | Retrospective | Looking back | "In retrospect, the judgment was..." |
   | Prospective | Looking forward | "Future experiments will determine..." |
   | Atemporal | Timeless | "Necessarily, if axioms then..." |

2. Training data balance:
   - Ensure each domain x judgment_type has examples in at least 3 aspectual frames
   - Test set includes novel frame combinations

3. Reward function invariance check:
   - Correct judgment across different framings of same proposition = bonus
   - Inconsistent judgment across frames = penalty

### Expected Impact
- **Forces operational invariance**: Same judgment regardless of aspectual framing
- **Prevents overfitting to surface form**: Model cannot rely on aspectual cues
- **Enriches coverage**: Multiplies effective training diversity

### Implementation Complexity
**Medium** - Requires aspectual tagging and rewriting of existing examples, plus balance checking in data generation.

---

## Insight 7: Domain-Specific Verification Chains

### Name
**Fulfillment Structure by Domain**

### Source
Lonergan's domain-specific judgment structures (mathematical: conceptual necessity; empirical: reversion to sense data; common sense: pragmatic; philosophic: performative self-consistency)

### Insight
The current training uses domain-appropriate vocabulary keywords, but does not enforce **domain-specific verification structure**. A mathematical judgment's fulfillment looks fundamentally different from an empirical judgment's fulfillment. **Domain-specific verification chain templates make this structure learnable.**

### Mechanism
1. Define verification chain templates by domain:

   **Mathematical**:
   ```
   Conditioned: [theorem/proposition]
   Link: If [axioms/definitions], then [conditioned]
   Fulfillment: By [proof steps], the link is satisfied
   Judgment: Yes (necessary) / No (contradictory) / Insufficient (unprovable)
   ```

   **Empirical**:
   ```
   Conditioned: [hypothesis about observables]
   Link: If [measurable conditions], then [conditioned]
   Fulfillment: Data shows [specific measurements]
   Judgment: Yes (verified) / No (falsified) / Insufficient (inconclusive)
   ```

   **Common Sense**:
   ```
   Conditioned: [practical claim about situation]
   Link: Based on [accumulated practical insights]
   Fulfillment: In this situation, [practical verification]
   Judgment: Yes (works) / No (fails) / Insufficient (untested)
   ```

   **Philosophic**:
   ```
   Conditioned: [position on knowing/being/objectivity]
   Link: If intelligently grasped and reasonably affirmed, then coherent
   Fulfillment: Affirming [position] while performing [cognitive acts] is [coherent/contradictory]
   Judgment: Position (invites development) / Counterposition (invites reversal)
   ```

2. Reward function checks domain-appropriate structure:
   - Mathematical judgment citing "observations" = penalty
   - Empirical judgment claiming "conceptual necessity" = penalty
   - Philosophic judgment without performative assessment = penalty

3. Training data explicitly models these templates

### Expected Impact
- **Operationalizes domain specificity**: Not just vocabulary but structure
- **Prevents domain conflation**: Mathematical reasoning cannot pose as empirical
- **Mirrors Lonergan's schema**: Training embodies the philosophical insight

### Implementation Complexity
**Medium** - Requires template-based generation, domain-specific structural validation in reward function, and schema update.

---

## Insight 8: Self-Similar Evaluation

### Name
**The Framework Judges Itself**

### Source
External framework's self-similar design; Lonergan's "the pattern of operations performed in discovering the pattern is not different from the pattern discovered"

### Insight
If the cognitive operations are truly invariant, **the evaluation of training success must itself exhibit the four-level pattern**. Currently, evaluation is external (binary accuracy). Self-similar evaluation means: (P1) attending to model outputs, (P2) understanding patterns in errors, (P3) judging training success, (P4) deciding next steps.

### Mechanism
1. Evaluation pipeline structured as cognitive operations:

   **P1-Evaluation (Attention)**:
   - Collect model outputs across test set
   - Present data without interpretation
   - Output: Raw performance data, output samples, failure cases

   **P2-Evaluation (Understanding)**:
   - Analyze patterns: What types of errors cluster?
   - Generate hypotheses: "Model conflates P2/P3 in philosophic domain"
   - Output: Hypothesis about operational learning state

   **P3-Evaluation (Judgment)**:
   - Check hypothesis against evidence
   - Determine: Is training succeeding? By what criteria?
   - Output: Judgment on training state with evidence

   **P4-Evaluation (Decision)**:
   - Based on judgment, decide: Continue? Adjust curriculum? Regenerate data?
   - Output: Concrete next action

2. Logging structure mirrors this:
   ```
   evaluation_log:
     attention:
       accuracy: 0.73
       samples: [...]
       error_distribution: {...}
     understanding:
       hypothesis: "P2/P3 conflation in philosophic domain"
       supporting_patterns: [...]
     judgment:
       training_succeeding: "Partial"
       evidence: "Mathematical/empirical strong, philosophic weak"
       conditions_for_success: [...]
       fulfillment_status: "Partially met"
     decision:
       action: "Generate more philosophic contrastive examples"
       rationale: "Address identified weakness"
   ```

3. The evaluation process becomes training data for meta-level operational learning

### Expected Impact
- **Self-consistency**: Framework models what it trains
- **Deeper diagnostics**: Four-level evaluation reveals more than accuracy
- **Virtuous circle**: If pattern is sound, evaluation improves pattern

### Implementation Complexity
**High** - Requires evaluation pipeline restructuring, logging schema, and potentially LLM-assisted analysis for P2-level pattern understanding.

---

## Insight 9: Dual-Packet Architecture for Role Separation

### Name
**Student-Evaluator Packet Decoupling**

### Source
External framework's dual-packet architecture (student packet: prompt, guidance, reference; evaluator packet: prompt, instructions, rubric)

### Insight
The current training conflates the "learner" role (model being trained) with the "evaluator" role (reward function). **Separating these into distinct information packets enables two-model training and cleaner reward signals.**

### Mechanism
1. Student packet (what the model sees during generation):
   ```json
   {
     "prompt": "Proposition: X. Domain: Empirical. Evaluate.",
     "guidance": "Identify conditions, assess fulfillment, render judgment",
     "scaffolds": ["[P3-JDG-XX]", "evidence_required"]
   }
   ```

2. Evaluator packet (what the reward function sees):
   ```json
   {
     "expected_judgment": "Yes",
     "rubric": {
       "must_include": [...],
       "unacceptable": [...],
       "partial_credit": {...}
     },
     "evidence_requirements": ["quote X", "cite measurement Y"],
     "distractor_patterns_to_penalize": ["P2_language", "authority_appeal"]
   }
   ```

3. The model never sees the evaluator packet during generation—only during training signal

4. Enables future two-model setup:
   - Model A generates (student)
   - Model B evaluates (trained as judge)
   - Cleaner reward signal, less reward hacking

### Expected Impact
- **Cleaner training signal**: Evaluator has full rubric, student has only guidance
- **Prevents reward hacking**: Model cannot reverse-engineer rubric from training
- **Enables judge training**: Can train a separate judge model on evaluator packets

### Implementation Complexity
**Low-Medium** - Primarily data schema restructuring. Two-model training is optional extension.

---

## Insight 10: Coverage Enforcement Matrix

### Name
**Multi-Dimensional Balance Verification**

### Source
External framework's systematic coverage (20 domains x 4 voices x 5 difficulties x 10 temporal frames)

### Insight
The current training has implicit domain variety but no explicit coverage enforcement. **A coverage matrix ensures systematic variation across all dimensions**, preventing clustering and ensuring operational learning generalizes.

### Mechanism
1. Define coverage dimensions:
   - **Domain**: Mathematical, Empirical, Common Sense, Pop Science, Philosophic (5)
   - **Judgment type**: Yes, No, Insufficient (3)
   - **Difficulty**: 1-5 (5)
   - **Aspectual frame**: Subset of 10 (choose 5 key frames)
   - **Distractor presence**: With/without contrastive examples (2)

2. Coverage matrix: 5 x 3 x 5 x 5 x 2 = 750 cells

3. Data generation targets:
   - Minimum 2 examples per cell = 1500 examples
   - Current 450 examples = ~60% of cells empty

4. Coverage report before training:
   ```
   Coverage Analysis:
   - Cells filled: 287/750 (38%)
   - Empty cells: 463
   - Most sparse: Philosophic x Insufficient x Level 5 (0 examples)
   - Most dense: Empirical x Yes x Level 2 (47 examples)
   ```

5. Targeted generation for sparse cells

### Expected Impact
- **Prevents clustering**: No domain/type/difficulty combination over-represented
- **Ensures generalization**: Model cannot rely on spurious correlations
- **Enables ablation**: Can test performance by dimension

### Implementation Complexity
**Low** - Primarily data analysis and targeted generation. No architectural changes.

---

## Summary: Priority Ranking

Based on expected impact and implementation complexity:

| Priority | Insight | Impact | Complexity | Rationale |
|----------|---------|--------|------------|-----------|
| 1 | Contrastive Distractor Learning | Very High | High | Addresses critical P2/P3 conflation |
| 2 | Evidence Grounding Requirement | High | Medium | Operationalizes Lonergan's core principle |
| 3 | Domain-Specific Verification Chains | High | Medium | Makes domain structure learnable |
| 4 | Rubric-Based Partial Credit | High | High | Dramatically improves learning signal |
| 5 | Coverage Enforcement Matrix | Medium | Low | Quick win for data quality |
| 6 | Graduated Difficulty Scaffolding | Medium | Medium | Enables curriculum learning |
| 7 | Operational Anchoring Tokens | Medium | Medium | Training scaffold, empirically testable |
| 8 | Temporal-Aspectual Variation | Medium | Medium | Forces operational invariance |
| 9 | Dual-Packet Architecture | Medium | Low-Medium | Clean separation, enables future extension |
| 10 | Self-Similar Evaluation | High | High | Philosophical consistency, deeper diagnostics |

---

## Implementation Roadmap

### Phase 1: Quick Wins (Week 1)
- Coverage Enforcement Matrix (Insight 10)
- Evidence Grounding Requirement (Insight 3)
- Domain-Specific Verification Chains (Insight 7)

### Phase 2: Core Improvements (Week 2-3)
- Contrastive Distractor Learning (Insight 2)
- Rubric-Based Partial Credit (Insight 5)
- Graduated Difficulty Scaffolding (Insight 4)

### Phase 3: Advanced Features (Week 4+)
- Operational Anchoring Tokens (Insight 1)
- Temporal-Aspectual Variation (Insight 6)
- Dual-Packet Architecture (Insight 9)
- Self-Similar Evaluation (Insight 8)

---

*These insights emerge from the intersection of external framework research and Lonergan's invariant cognitive pattern. Implementation should be guided by empirical results—each insight is a hypothesis to be tested, not a dogma to be followed.*
