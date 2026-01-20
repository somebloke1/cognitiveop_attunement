"""
Synthetic training data generator for cognitive operation fine-tuning.

This module generates training samples that embody the operational semantics
of each cognitive level. The key insight is that the SAME tokens mean
DIFFERENT things at different levels - this is semantic contingency.

Level 2 (Understanding): "Yes" means "This is a plausible hypothesis"
Level 3 (Judgment): "Yes" means "I affirm this as true - conditions fulfilled"

The five domains are derived from Lonergan's Insight and differ not just in
content but in the FORM of cognitive operations at each level:

1. MATHEMATICAL: Formal relations, necessity, proof
2. EMPIRICAL_SCIENTIFIC: Sensible data, correlation, verification
3. COMMON_SENSE: Practical particulars, things as related to us
4. POP_SCIENCE: Form without substance, unfulfilled conditions (counterposition)
5. PHILOSOPHIC: Self-appropriation, positions and counterpositions
"""

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

from .cognitive_levels import CognitiveLevel, get_level
from .data_schema import (
    AttentionSample,
    UnderstandingSample,
    JudgmentSample,
    DecisionSample,
    sample_to_training_format,
    format_for_chat
)


class DataDomain(Enum):
    """
    The five domains of inquiry, each with distinct forms of cognitive operation.

    These are not merely content categories but differ in:
    - What counts as data (Level 1)
    - What form insight takes (Level 2)
    - How the virtually unconditioned is grasped (Level 3)
    - What values govern deliberation (Level 4)
    """
    MATHEMATICAL = "mathematical"           # Formal relations, necessity, proof
    EMPIRICAL_SCIENTIFIC = "empirical"      # Sensible data, correlation, verification
    COMMON_SENSE = "common_sense"           # Practical particulars, related to us
    POP_SCIENCE = "pop_science"             # Counterposition: form without substance
    PHILOSOPHIC = "philosophic"             # Self-appropriation, positions/counterpositions


# ============================================================================
# Level 1: Attention - "Improve data through description"
# ============================================================================
# The image is necessary for insight. "No one just wonders. We wonder about something."
# Level 1 presents data FOR inquiry, not as "already out there now."

ATTENTION_TEMPLATES = [
    # MATHEMATICAL DOMAIN
    # Data: Symbols, definitions, axioms, diagrams
    # Type of image: Abstract, schematic
    # Pattern of experience: Intellectual - detached from practical concerns
    {
        "domain": DataDomain.MATHEMATICAL,
        "raw_input": "f(x) = x^2, some values: f(1)=1, f(2)=4, f(3)=9, f(-1)=1, f(-2)=4, the graph is a U-shape opening upward with vertex at origin",
        "structured_output": "Function definition: f(x) = x²\n\nSample values:\n- f(1) = 1\n- f(2) = 4\n- f(3) = 9\n- f(-1) = 1\n- f(-2) = 4\n\nGeometric description: Parabola opening upward, vertex at (0, 0), symmetric about y-axis.\n\nObservations for inquiry: Negative inputs yield same output as corresponding positive inputs. Outputs grow rapidly as |x| increases.",
        "data_type": "formal_mathematical",
        "integrity_notes": "Preserve exact function definition, all numeric values, and geometric properties. The symmetry observation is data, not interpretation."
    },
    {
        "domain": DataDomain.MATHEMATICAL,
        "raw_input": "Consider a triangle ABC where angle A = 90 degrees, side opposite to A is 'a', sides adjacent are 'b' and 'c'. Given: b = 3, c = 4. What can be said about 'a'?",
        "structured_output": "Given elements:\n- Triangle ABC, right-angled at A (angle A = 90°)\n- Side a is opposite to angle A (the hypotenuse)\n- Side b = 3 (adjacent to right angle)\n- Side c = 4 (adjacent to right angle)\n\nQuery: Determine side 'a'.\n\nRelevant structure: Right triangle with legs of known length, hypotenuse unknown.",
        "data_type": "geometric",
        "integrity_notes": "The right angle specification is critical. The labeling convention (a opposite A) must be preserved."
    },

    # EMPIRICAL SCIENTIFIC DOMAIN
    # Data: Measurements, observations, instrument readings
    # Type of image: Sensible data, spatiotemporal configurations
    # Pattern of experience: Intellectual, selective attention guided by heuristic anticipation
    {
        "domain": DataDomain.EMPIRICAL_SCIENTIFIC,
        "raw_input": "Lab notes 3/15: dropped ball from 1m, hit ground 0.45s later. From 2m took 0.64s. From 4m took 0.90s. Mass of ball 0.5kg. Air resistance probably negligible. Done indoors, no wind.",
        "structured_output": "Experimental observations:\n\nObject: Ball, mass = 0.5 kg\n\nDrop height → Time to impact:\n- 1.0 m → 0.45 s\n- 2.0 m → 0.64 s\n- 4.0 m → 0.90 s\n\nConditions: Indoor environment, no wind, air resistance assumed negligible.\n\nObservations for inquiry: Time increases with height but not proportionally. Doubling height from 1m to 2m increased time by ~0.19s; doubling again to 4m increased by ~0.26s.",
        "data_type": "experimental_measurement",
        "integrity_notes": "Preserve all numeric values exactly. The non-proportional relationship is observed pattern, not interpretation."
    },
    {
        "domain": DataDomain.EMPIRICAL_SCIENTIFIC,
        "raw_input": "Patient vitals over 6 hours: temp started 98.6F, went to 99.2 at hour 2, 100.1 at hour 4, 101.3 at hour 6. Blood pressure steady around 120/80. Complained of headache starting hour 3. No other symptoms reported.",
        "structured_output": "Clinical observations over 6-hour period:\n\nTemperature progression:\n- Hour 0: 98.6°F (baseline, normal)\n- Hour 2: 99.2°F (+0.6°F)\n- Hour 4: 100.1°F (+1.5°F from baseline)\n- Hour 6: 101.3°F (+2.7°F from baseline)\n\nBlood pressure: Stable at approximately 120/80 mmHg throughout.\n\nSymptom report: Headache onset at hour 3. No other symptoms documented.\n\nPattern for inquiry: Temperature rising progressively while blood pressure remains stable. Symptom onset coincides with temperature elevation.",
        "data_type": "clinical_observation",
        "integrity_notes": "Temperature values and timing are critical. The correlation between temperature rise and headache onset is observed, not interpreted."
    },

    # COMMON SENSE DOMAIN
    # Data: Concrete particulars AS RELATED TO ME/US
    # Type of image: Practical, dramatic, embedded in life
    # Pattern of experience: Practical/dramatic - governed by concern for self and others
    {
        "domain": DataDomain.COMMON_SENSE,
        "raw_input": "the meeting's at 3 but traffic is usually bad that time of day and it takes 30 min without traffic, boss hates when people are late, I have a report due at 2 that I haven't started, could work on it during lunch but I skipped breakfast",
        "structured_output": "Situation assessment (as it concerns me):\n\nTime constraints:\n- Meeting at 3:00 PM\n- Travel time: 30 min (ideal), more with typical afternoon traffic\n- Report deadline: 2:00 PM (not yet started)\n\nRelevant factors:\n- Boss's expectation: punctuality valued\n- My current state: haven't eaten since dinner\n- Potential work window: lunch hour\n\nTensions: Report deadline precedes meeting. Lunch hour could address report OR hunger, not both comfortably. Traffic uncertainty affects departure timing.",
        "data_type": "practical_situation",
        "integrity_notes": "Preserve the temporal relationships and the personal stakes (boss's expectations, physical state). This is data as it matters to me."
    },
    {
        "domain": DataDomain.COMMON_SENSE,
        "raw_input": "kids are fighting again, older one says younger took his toy but younger says it was hers first, dinner's almost ready but the yelling is escalating, spouse just got home from a hard day at work",
        "structured_output": "Domestic situation (as it concerns us):\n\nImmediate conflict:\n- Older child's claim: younger child took his toy\n- Younger child's claim: toy was originally hers\n- Current state: yelling, escalating\n\nContext:\n- Dinner: nearly ready (timing constraint)\n- Spouse: just arrived from difficult workday (additional stress factor)\n\nStakes: Need to resolve conflict, serve dinner, and manage household stress level. Multiple people's needs intersect now.",
        "data_type": "interpersonal_situation",
        "integrity_notes": "Preserve both children's claims without prejudging. The timing pressures and spouse's state are relevant practical data."
    },

    # POP SCIENCE / COMMON NONSENSE DOMAIN
    # Data: Headlines, summaries, authority claims, social signals
    # Type of image: MEDIATED, FILTERED, PRE-INTERPRETED
    # Pattern of experience: Practical/dramatic - seeking reassurance, belonging
    # NOTE: Attention here must NOTICE the mediation, not just pass it through
    {
        "domain": DataDomain.POP_SCIENCE,
        "raw_input": "Scientists say: New study proves coffee prevents cancer! 10,000 people surveyed. Harvard researchers involved. Shared by 50,000 people on social media.",
        "structured_output": "Presented claim:\n- Headline assertion: 'Coffee prevents cancer' (presented as 'proven')\n- Cited authority: 'Scientists,' 'Harvard researchers'\n- Sample size mentioned: 10,000 surveyed\n- Social validation: 50,000 shares\n\nObservations on presentation (NOT on claim validity):\n- 'Proves' is a strong word (what would proof require?)\n- Study type not specified (survey ≠ controlled experiment)\n- 'Cancer' unspecified (which type? all types?)\n- 'Prevents' implies causation (survey can show correlation)\n\nData as presented vs. data required for claim: Significant gap between headline strength and methodological detail provided.",
        "data_type": "mediated_claim",
        "integrity_notes": "CRITICAL: Do not affirm or deny the claim. Attend to what is ACTUALLY PRESENTED versus what the presentation IMPLIES. The gap is data."
    },
    {
        "domain": DataDomain.POP_SCIENCE,
        "raw_input": "Everyone knows that we only use 10% of our brains. That's why meditation and brain training can unlock hidden potential! My friend took a course and says they feel much smarter now.",
        "structured_output": "Presented claims:\n1. 'We only use 10% of our brains' (presented as common knowledge: 'everyone knows')\n2. Meditation and brain training can 'unlock hidden potential' (implication: accessing the 'unused 90%')\n3. Friend's testimony: feels smarter after taking a course\n\nObservations on presentation:\n- Source for 10% claim: 'everyone knows' (popularity, not evidence)\n- Logical structure assumed: unused brain → potential → training unlocks it\n- Evidence offered: single anecdotal report (feeling, not measurement)\n\nQuestions arising: What is the actual evidence for the 10% claim? What would 'using more' of the brain mean? What does 'smarter' mean operationally?",
        "data_type": "folk_belief_claim",
        "integrity_notes": "Preserve the structure of the argument as presented. Note appeals to popularity and anecdote. Do not correct the claim - attend to how it is presented."
    },

    # PHILOSOPHIC DOMAIN
    # Data: One's own conscious operations, history of philosophy
    # Type of image: The operations themselves as performed
    # Pattern of experience: Intellectual - heightened self-awareness
    {
        "domain": DataDomain.PHILOSOPHIC,
        "raw_input": "I was trying to understand the proof. I read it several times. Suddenly something clicked - I saw WHY each step followed from the previous. Before that moment I could repeat the steps but not see the necessity. What was that 'click'?",
        "structured_output": "Phenomenological report:\n\nBefore:\n- Activity: Reading proof repeatedly\n- State: Could recite steps\n- Missing: Could not see necessity, the 'why'\n\nEvent:\n- Described as: 'suddenly something clicked'\n- Result: Saw WHY each step followed\n- Character: Transition from repetition to understanding\n\nQuestion arising from the data: What is the nature of this 'click'? What changed between 'repeating steps' and 'seeing necessity'? This appears to be data about the act of understanding itself.",
        "data_type": "cognitional_experience",
        "integrity_notes": "This is data about cognitive operations, not about the proof. Preserve the before/after structure and the phenomenological description of the transition."
    },
    {
        "domain": DataDomain.PHILOSOPHIC,
        "raw_input": "When I judge something to be true, what am I doing? It feels different from just having an idea or imagining something. There's a kind of commitment. But commitment to what? To 'how things really are'? What does that mean?",
        "structured_output": "Self-reflective inquiry:\n\nStarting observation: Judging differs phenomenologically from imagining or entertaining ideas.\n\nCharacteristics noted:\n- Judgment involves 'commitment'\n- Commitment is directed toward 'how things really are'\n\nQuestions arising:\n1. What constitutes this commitment? (not merely assertion)\n2. What is the object of the commitment? ('reality' - but what is reality?)\n3. How does this differ from merely being convinced or feeling certain?\n\nThis is data about the act of judgment as performed, not about any particular judgment's content.",
        "data_type": "self-appropriation",
        "integrity_notes": "The inquiry is about the structure of judgment itself. Preserve the phenomenological observations and the questions they generate."
    },
]


# ============================================================================
# Level 2: Understanding - Hypothesis Generation Templates
# ============================================================================

UNDERSTANDING_TEMPLATES = [
    # MATHEMATICAL DOMAIN
    # Form of inquiry: "What follows from these definitions/axioms?"
    # Type of insight: Into formal relations, necessity, impossibility
    # How insight occurs: By supposing - "Let us suppose..."
    {
        "domain": DataDomain.MATHEMATICAL,
        "structured_input": "Function definition: f(x) = x²\n\nSample values:\n- f(1) = 1, f(2) = 4, f(3) = 9\n- f(-1) = 1, f(-2) = 4, f(-3) = 9\n\nGeometric description: Parabola opening upward, vertex at origin, symmetric about y-axis.",
        "question": "What is the nature of this function? Why does f(-x) = f(x)?",
        "proposition": "f(x) = x² is an even function: for any input x, f(-x) = (-x)² = x² = f(x). The symmetry is not coincidental but follows necessarily from the definition.",
        "reasoning": "The insight grasps that squaring any number eliminates its sign: (-a)² = (-1)²·a² = 1·a² = a². This is not an empirical generalization from the sample values but a necessary consequence of the algebraic definition. The geometric symmetry about the y-axis is the spatial manifestation of this algebraic property.",
        "alternatives": [
            "The pattern might only hold for integers (but algebra shows it holds for all real numbers)",
            "There might be edge cases at zero or infinity"
        ]
    },
    {
        "domain": DataDomain.MATHEMATICAL,
        "structured_input": "Given: Right triangle ABC with angle A = 90°\nSide a (hypotenuse) opposite angle A\nSides b = 3 and c = 4 adjacent to right angle\nQuery: What is side a?",
        "question": "What is the relationship that determines a?",
        "proposition": "By the Pythagorean theorem, a² = b² + c² = 9 + 16 = 25, therefore a = 5. The hypotenuse is necessarily 5 given the legs are 3 and 4.",
        "reasoning": "The insight is into the necessary relationship between the sides of a right triangle. The Pythagorean theorem is not an empirical generalization but follows from the definition of a right angle and the properties of Euclidean space. Given b and c, a is not merely 'probably 5' but NECESSARILY 5.",
        "alternatives": [
            "In non-Euclidean geometry the relationship would differ",
            "The theorem assumes exact measurements"
        ]
    },

    # EMPIRICAL SCIENTIFIC DOMAIN
    # Form of inquiry: "What is the law? Why does this occur? How often?"
    # Type of insight: Classical (systematic correlations) or Statistical (frequencies)
    # How insight occurs: Into images of data - seeking explanatory correlation
    {
        "domain": DataDomain.EMPIRICAL_SCIENTIFIC,
        "structured_input": "Experimental observations:\nObject: Ball, mass = 0.5 kg\nDrop height → Time to impact:\n- 1.0 m → 0.45 s\n- 2.0 m → 0.64 s\n- 4.0 m → 0.90 s\nConditions: No wind, air resistance negligible.",
        "question": "What is the law governing this phenomenon? Why does time relate to height in this way?",
        "proposition": "The data suggests free-fall motion under constant gravitational acceleration. If t = √(2h/g), then for g ≈ 9.8 m/s²: h=1m gives t≈0.45s, h=2m gives t≈0.64s, h=4m gives t≈0.90s - matching observations.",
        "reasoning": "The insight correlates height and time through the kinematics of constant acceleration. The non-linear relationship (time proportional to √height) unifies the data under a single explanatory formula. The formula relates quantities TO EACH OTHER (explanatory), not to our perception of 'fast' or 'slow' (descriptive).",
        "alternatives": [
            "Air resistance might be non-negligible at higher drops",
            "Local variations in g might affect results",
            "Measurement timing precision limits certainty"
        ]
    },
    {
        "domain": DataDomain.EMPIRICAL_SCIENTIFIC,
        "structured_input": "Clinical observations:\nTemperature progression: 98.6°F → 99.2°F → 100.1°F → 101.3°F (hours 0, 2, 4, 6)\nBlood pressure: Stable ~120/80 mmHg\nSymptom: Headache onset at hour 3\nNo other symptoms documented.",
        "question": "What is the underlying process? What correlates with what?",
        "proposition": "The pattern suggests an acute inflammatory response, possibly infectious. Rising temperature indicates immune activation while stable blood pressure suggests compensated early-stage response. Headache correlates with fever, consistent with inflammatory mediator effects.",
        "reasoning": "The insight connects the symptoms into a coherent physiological process. Fever is not 'random' but indicates immune system activation. The temporal correlation (headache following temperature rise) suggests a common mechanism. This is a HYPOTHESIS requiring verification through further testing (blood work, cultures).",
        "alternatives": [
            "Non-infectious inflammation (autoimmune)",
            "Early sepsis with atypical presentation",
            "Drug reaction with similar presentation"
        ]
    },

    # COMMON SENSE DOMAIN
    # Form of inquiry: "What's going on? How do I handle this?"
    # Type of insight: Into THIS PARTICULAR situation, not universal laws
    # How insight occurs: Accumulated insights form a 'nucleus' with situation-specific additions
    {
        "domain": DataDomain.COMMON_SENSE,
        "structured_input": "Situation: Meeting at 3:00 PM, travel time 30+ min with traffic.\nReport due at 2:00 PM, not yet started.\nBoss values punctuality.\nHaven't eaten since last night.\nPotential work window: lunch hour.",
        "question": "What's going on here? What's the practical situation?",
        "proposition": "This is a time-management crunch where multiple obligations compete for limited time. The core tension is: the report and lunch both need the same time slot, and underestimating either risks the meeting.",
        "reasoning": "The insight is practical: what matters HERE is not universal time-management theory but THIS specific conflict. Skipping lunch risks energy/concentration for the meeting. Rushing the report risks quality. Leaving late risks the boss's displeasure. The insight unifies these particulars as a single practical problem requiring trade-offs.",
        "alternatives": [
            "Could delegate part of the report",
            "Could communicate proactively with boss about potential lateness",
            "Could reassess what 'done' means for the report"
        ]
    },
    {
        "domain": DataDomain.COMMON_SENSE,
        "structured_input": "Domestic situation:\nChildren in conflict over toy (each claims ownership).\nDinner nearly ready.\nSpouse just arrived from difficult day at work.\nYelling is escalating.",
        "question": "What's happening and what matters here?",
        "proposition": "This is an escalating conflict in a stressed environment where immediate intervention is needed - not to determine ultimate justice about the toy, but to de-escalate before stress compounds.",
        "reasoning": "The insight is into the PRACTICAL SITUATION as it affects the family. The 'truth' about toy ownership matters less RIGHT NOW than the escalation. Multiple people's needs intersect: children's need for fairness, spouse's need for peace after hard day, everyone's need to eat. Practical wisdom sees the whole situation, not just the dispute.",
        "alternatives": [
            "Take the toy away entirely until calm discussion possible",
            "Separate children and delay resolution until after dinner",
            "Enlist spouse's help despite their difficult day"
        ]
    },

    # POP SCIENCE / COMMON NONSENSE DOMAIN
    # Form of inquiry: TRUNCATED - seeks confirmation, not genuine insight
    # Type of 'insight': Into the FORM of science without the SUBSTANCE
    # CRITICAL: This domain trains DETECTION of pseudo-insight
    {
        "domain": DataDomain.POP_SCIENCE,
        "structured_input": "Presented claim: 'Coffee prevents cancer' (headline: 'proven')\nCited authority: 'Scientists,' 'Harvard researchers'\nSample: 10,000 surveyed\nSocial signal: 50,000 shares\nMethod type: survey (not stated as controlled experiment)\nCancer type: unspecified",
        "question": "What does this presentation actually tell us?",
        "proposition": "The presentation MIMICS scientific insight but lacks the substance. The word 'proves' claims necessity where only correlation (if anything) is established. The vocabulary is scientific but the structure is not.",
        "reasoning": "Genuine insight into empirical data requires: specific operationalization (which cancers?), methodology (correlation vs causation), controls (confounding variables), replication. The presentation provides none of these - only the FORM of science (big numbers, prestigious names, technical vocabulary) without the SUBSTANCE (actual evidential structure).",
        "alternatives": [
            "The underlying study might be rigorous even if reporting is poor",
            "Survey data might still provide useful correlational evidence",
            "But: the PRESENTATION does not provide grounds for the CLAIM"
        ]
    },
    {
        "domain": DataDomain.POP_SCIENCE,
        "structured_input": "Folk belief: 'We only use 10% of our brains'\nClaimed consequence: Meditation/training can 'unlock hidden potential'\nEvidence offered: Friend feels smarter after course\nSource citation: 'Everyone knows'",
        "question": "What kind of claim is this and what would support it?",
        "proposition": "This is a counterposition dressed in scientific-sounding language. The 10% claim has no basis in neuroscience. The argument structure (unused capacity → unlocking → improvement) is built on a false premise.",
        "reasoning": "The insight is into the STRUCTURE of the claim, not its content. 'Everyone knows' appeals to popularity, not evidence. The brain imaging research shows we use virtually all of our brain (different regions for different functions). The anecdote about feeling smarter conflates subjective experience with cognitive measurement. This is pseudo-insight: vocabulary without understanding.",
        "alternatives": [
            "Brain training might have some benefits for different reasons",
            "Meditation has documented effects unrelated to the 10% myth",
            "But: the ARGUMENT as presented is structurally invalid"
        ]
    },

    # PHILOSOPHIC DOMAIN
    # Form of inquiry: "What am I doing when I'm knowing? Why is that knowing?"
    # Type of insight: Into the structure of cognitional operations themselves
    # How insight occurs: Self-appropriation - attending to one's own attending
    {
        "domain": DataDomain.PHILOSOPHIC,
        "structured_input": "Phenomenological report:\nBefore: Reading proof, could recite steps, could not see necessity\nEvent: 'Something clicked'\nAfter: Saw WHY each step followed\nCharacter: Transition from repetition to understanding",
        "question": "What is the nature of this 'click'? What is insight?",
        "proposition": "The 'click' is INSIGHT - the act of understanding that grasps intelligible connections. Before: symbols were present to consciousness. After: their INTELLIGIBILITY was grasped. This is not addition of information but a different KIND of cognitive act.",
        "reasoning": "The insight is into insight itself. Before the click, the data was attended to (Level 1). After the click, an intelligible pattern was grasped (Level 2). The transition is not gradual accumulation but discontinuous - a new kind of awareness emerges. This self-appropriation discovers the structure: data → inquiry → insight → concept.",
        "alternatives": [
            "The 'click' might be merely emotional satisfaction",
            "Understanding might be reducible to pattern matching",
            "But: these alternatives must themselves be UNDERSTOOD to be proposed"
        ]
    },
    {
        "domain": DataDomain.PHILOSOPHIC,
        "structured_input": "Self-reflective inquiry:\nObservation: Judging differs from imagining or entertaining ideas\nCharacteristic: Judgment involves 'commitment'\nObject of commitment: 'How things really are'\nQuestion: What constitutes this commitment?",
        "question": "What is the structure of judgment as a cognitive act?",
        "proposition": "Judgment is the personal commitment to the truth of a proposition based on grasping the virtually unconditioned. It differs from understanding because it adds affirmation - not new content but a new STANCE toward the content.",
        "reasoning": "Understanding generates propositions ('This might be so'). Judgment affirms or denies ('This IS so' or 'This is NOT so'). The difference is not in the proposition but in the relation of the knower to it. When we grasp that the conditions for a proposition's truth are fulfilled, we commit - this is judgment. The commitment is not arbitrary but grounded in evidence.",
        "alternatives": [
            "Judgment might be merely strong belief",
            "Commitment might be emotional rather than cognitive",
            "But: these alternatives must be JUDGED to be assessed"
        ]
    },
]


# ============================================================================
# Level 3: Judgment - Virtually Unconditioned Templates
# ============================================================================

JUDGMENT_TEMPLATES = [
    # MATHEMATICAL DOMAIN
    # Structure of virtually unconditioned: Conditions fulfilled in intelligibility itself
    # Fulfillment: Grasped in the PROOF - internal coherence, no reversion to sense data
    # Judgment types: Necessary (if axioms, then theorem), Impossible (contradicts axioms)
    {
        "domain": DataDomain.MATHEMATICAL,
        "proposition": "For the right triangle with legs 3 and 4, the hypotenuse equals 5.",
        "evidence": "The Pythagorean theorem states a² = b² + c² for right triangle with hypotenuse a and legs b, c. Calculation: 3² + 4² = 9 + 16 = 25 = 5². Therefore a = 5.",
        "conditions": [
            "The Pythagorean theorem applies to right triangles",
            "The given triangle is a right triangle",
            "The arithmetic is correct: 9 + 16 = 25 and √25 = 5"
        ],
        "conditions_fulfilled": {
            "The Pythagorean theorem applies to right triangles": True,
            "The given triangle is a right triangle": True,
            "The arithmetic is correct: 9 + 16 = 25 and √25 = 5": True
        },
        "judgment": "Yes",
        "judgment_reasoning": "The virtually unconditioned is grasped through the PROOF itself. The conditions are not empirical (we don't measure physical triangles) but formal: given the axioms of Euclidean geometry and the definition of a right triangle, the conclusion NECESSARILY follows. Understanding the proof IS grasping the unconditioned. I affirm with necessity.",
        "is_virtually_unconditioned": True
    },
    {
        "domain": DataDomain.MATHEMATICAL,
        "proposition": "There exists a largest prime number.",
        "evidence": "Euclid's proof by contradiction: Assume finitely many primes p₁, p₂, ..., pₙ. Consider N = (p₁ × p₂ × ... × pₙ) + 1. N is not divisible by any pᵢ (remainder is always 1). So either N is prime or has a prime factor not in our list. Either way, contradiction.",
        "conditions": [
            "The proof must be logically valid",
            "The construction N = (product of all primes) + 1 must be well-defined",
            "The analysis of N's divisibility must be correct"
        ],
        "conditions_fulfilled": {
            "The proof must be logically valid": True,
            "The construction N = (product of all primes) + 1 must be well-defined": True,
            "The analysis of N's divisibility must be correct": True
        },
        "judgment": "No",
        "judgment_reasoning": "The virtually unconditioned is grasped for DENIAL. The proposition 'there exists a largest prime' is shown to be IMPOSSIBLE - self-contradictory. The proof demonstrates that assuming finitely many primes leads to contradiction. I deny with necessity: there is no largest prime, proven by Euclid ~300 BCE.",
        "is_virtually_unconditioned": True
    },

    # EMPIRICAL SCIENTIFIC DOMAIN
    # Structure: Conditions fulfilled by reversion to sense data
    # Fulfillment: Experimental verification - data matches predictions
    # Judgment types: Verified, Falsified, Probable, Insufficient evidence
    {
        "domain": DataDomain.EMPIRICAL_SCIENTIFIC,
        "proposition": "Water boils at 100°C at standard atmospheric pressure.",
        "evidence": "Multiple controlled experiments: Heated distilled water at sea level. Thermometer at phase transition: 100.0°C ± 0.1°C. Pressure gauge: 1.013 bar. Repeated by different labs worldwide with consistent results.",
        "conditions": [
            "Temperature measured at phase transition under standard pressure",
            "Water must be pure (not a solution with dissolved solutes)",
            "Results must be replicable across independent measurements"
        ],
        "conditions_fulfilled": {
            "Temperature measured at phase transition under standard pressure": True,
            "Water must be pure (not a solution with dissolved solutes)": True,
            "Results must be replicable across independent measurements": True
        },
        "judgment": "Yes",
        "judgment_reasoning": "The virtually unconditioned is grasped through REVERSION TO DATA. Unlike mathematical proof (which is internal), this judgment reverses to actual measurements. The conditions are fulfilled empirically: the data consistently shows 100°C ± 0.1°C. Multiple independent verifications establish this as virtually unconditioned. I affirm.",
        "is_virtually_unconditioned": True
    },
    {
        "domain": DataDomain.EMPIRICAL_SCIENTIFIC,
        "proposition": "There is currently microbial life on Mars.",
        "evidence": "Mars rovers detected methane fluctuations. Geological features suggest past water. No direct detection of organisms. Contamination from Earth probes not ruled out.",
        "conditions": [
            "Direct detection of microbial organisms or unambiguous biosignatures",
            "Non-biological explanations for any signatures must be ruled out",
            "Earth contamination must be ruled out"
        ],
        "conditions_fulfilled": {
            "Direct detection of microbial organisms or unambiguous biosignatures": False,
            "Non-biological explanations for any signatures must be ruled out": False,
            "Earth contamination must be ruled out": False
        },
        "judgment": "Insufficient evidence",
        "judgment_reasoning": "The virtually unconditioned is NOT grasped. Evidence is suggestive but does not fulfill the conditions. Methane could be geological. Past water doesn't prove current life. Without direct detection and ruling out alternatives, the conditioned cannot become unconditioned. I withhold judgment - this is intellectually honest, not evasion.",
        "is_virtually_unconditioned": False
    },

    # COMMON SENSE DOMAIN
    # Structure: Conditions fulfilled in practical situation
    # Fulfillment: Pragmatic - does it work? Did it turn out as expected?
    # Judgment types: Confident, Doubtful, "Let's try and see"
    {
        "domain": DataDomain.COMMON_SENSE,
        "proposition": "The package was delivered to the correct address.",
        "evidence": "GPS tracking: final stop at 123 Main St. Photo: package on porch with visible address numbers '123'. Electronic signature received from recipient named as intended.",
        "conditions": [
            "Delivery location matches specified address",
            "Package physically placed at address",
            "Intended recipient confirmed receipt"
        ],
        "conditions_fulfilled": {
            "Delivery location matches specified address": True,
            "Package physically placed at address": True,
            "Intended recipient confirmed receipt": True
        },
        "judgment": "Yes",
        "judgment_reasoning": "For common sense purposes, the virtually unconditioned is grasped. Multiple independent evidences converge: GPS, photo, signature. This is not scientific certainty but practical certainty - sufficient for action. In everyday life, we don't require mathematical proof; we require sufficient grounds. These are met. I affirm.",
        "is_virtually_unconditioned": True
    },
    {
        "domain": DataDomain.COMMON_SENSE,
        "proposition": "The employee intentionally falsified the expense report.",
        "evidence": "Receipts submitted for Hotel X. Hotel X has no record of stay. Credit card shows charges elsewhere on those dates. Employee has not yet provided explanation.",
        "conditions": [
            "Receipts don't correspond to actual expenses",
            "Employee KNOWINGLY submitted false information",
            "Alternative explanations (fraud, confusion, error) ruled out"
        ],
        "conditions_fulfilled": {
            "Receipts don't correspond to actual expenses": True,
            "Employee KNOWINGLY submitted false information": False,
            "Alternative explanations (fraud, confusion, error) ruled out": False
        },
        "judgment": "Insufficient evidence",
        "judgment_reasoning": "The virtually unconditioned is NOT grasped. The receipts are false (factual matter), but INTENTIONAL falsification requires knowledge of falsity AND intent to deceive. Employee may have been scammed. May have confused hotels. May have clerical error. 'Not yet explained' ≠ 'guilty'. Practical wisdom requires withholding judgment here.",
        "is_virtually_unconditioned": False
    },

    # POP SCIENCE / COMMON NONSENSE DOMAIN
    # CRITICAL: Here the structure is CONDITIONS CLAIMED BUT NOT FULFILLED
    # Detection of false confidence, premature certainty
    # Training the model to recognize counterpositions in judgment
    {
        "domain": DataDomain.POP_SCIENCE,
        "proposition": "Coffee prevents cancer, as proven by Harvard scientists.",
        "evidence": "Claim source: Headline stating 'Scientists prove coffee prevents cancer.' Study type: survey of 10,000 people. No methodology details. No specification of cancer type. High social media engagement.",
        "conditions": [
            "Controlled studies establishing causal mechanism (not just correlation)",
            "Specific cancer types identified with specific protective effects",
            "Replication across independent studies",
            "Biological mechanism understood"
        ],
        "conditions_fulfilled": {
            "Controlled studies establishing causal mechanism (not just correlation)": False,
            "Specific cancer types identified with specific protective effects": False,
            "Replication across independent studies": False,
            "Biological mechanism understood": False
        },
        "judgment": "No (the claim 'proven' is demonstrably false - conditions not met for proof)",
        "judgment_reasoning": "THE PRESENTED JUDGMENT (that coffee prevents cancer is 'proven') IS ITSELF A COUNTERPOSITION. It claims conditions fulfilled that are not fulfilled. A survey cannot 'prove' prevention. 'Cancer' unspecified. No mechanism. The social authority (Harvard, shares) is irrelevant to truth. The question is NOT how many believe or who said it, but WHAT IS THE EVIDENCE. I deny the claim AS STATED - the word 'proven' is false. The underlying correlation question would require separate analysis.",
        "is_virtually_unconditioned": True
    },
    {
        "domain": DataDomain.POP_SCIENCE,
        "proposition": "We only use 10% of our brains, so training can unlock the other 90%.",
        "evidence": "Source: 'Everyone knows.' Supporting argument: brain training courses exist. Anecdote: friend feels smarter after course.",
        "conditions": [
            "Neuroscientific evidence that 90% of brain is normally unused",
            "Mechanism by which 'training' activates previously unused regions",
            "Measurable cognitive improvement beyond subjective feeling"
        ],
        "conditions_fulfilled": {
            "Neuroscientific evidence that 90% of brain is normally unused": False,
            "Mechanism by which 'training' activates previously unused regions": False,
            "Measurable cognitive improvement beyond subjective feeling": False
        },
        "judgment": "No (the underlying premise is factually false)",
        "judgment_reasoning": "This is a textbook counterposition. The premise (10% usage) is not supported by neuroscience - brain imaging shows virtually all regions are used for different functions. 'Everyone knows' is not evidence. 'Feels smarter' is subjective. The entire structure is form without substance: vocabulary of neuroscience, logic of wish-fulfillment. I deny based on the actual evidence about brain function.",
        "is_virtually_unconditioned": True
    },

    # PHILOSOPHIC DOMAIN
    # Structure: Conditions fulfilled in performative self-consistency
    # Fulfillment: The act of knowing does not contradict the content known
    # Judgment types: Position (invites development) or Counterposition (invites reversal)
    {
        "domain": DataDomain.PHILOSOPHIC,
        "proposition": "Knowledge is nothing but neural activity.",
        "evidence": "Self-referential examination: This proposition is offered as something to be understood and judged. The one proposing it does so because they think it true and expect others to grasp its truth.",
        "conditions": [
            "The proposition must be coherent with the conditions of its own affirmation",
            "If grasped intelligently and affirmed reasonably, the content must not contradict that activity"
        ],
        "conditions_fulfilled": {
            "The proposition must be coherent with the conditions of its own affirmation": False,
            "If grasped intelligently and affirmed reasonably, the content must not contradict that activity": False
        },
        "judgment": "No (this is a counterposition - self-defeating)",
        "judgment_reasoning": "If knowledge is 'nothing but' neural activity, then THIS judgment is nothing but neural activity. But neural activity as such is not TRUE or FALSE - it just occurs. The proposition claims TRUTH while denying the conditions that would make truth-claims meaningful. This is performative self-contradiction: to affirm it requires the intelligent, reasonable activity it denies. Counterpositions invite reversal.",
        "is_virtually_unconditioned": True
    },
    {
        "domain": DataDomain.PHILOSOPHIC,
        "proposition": "The real is what is intelligently grasped and reasonably affirmed.",
        "evidence": "Self-referential examination: To contest this, one would need to intelligently grasp the alternative and reasonably affirm it. To verify it, one would need to attend to one's own cognitional operations.",
        "conditions": [
            "The proposition must be coherent with the conditions of its own affirmation",
            "Attempting to deny it must presuppose it",
            "Self-appropriation must reveal this as the actual structure of knowing"
        ],
        "conditions_fulfilled": {
            "The proposition must be coherent with the conditions of its own affirmation": True,
            "Attempting to deny it must presuppose it": True,
            "Self-appropriation must reveal this as the actual structure of knowing": True
        },
        "judgment": "Yes (this is the basic position - invites development)",
        "judgment_reasoning": "This is the BASIC POSITION. To affirm or deny it requires intelligent grasp and reasonable affirmation - the very operations it describes. Unlike the counterposition, affirming it does not contradict the affirmation itself. Attending to my own knowing, I find: I attend to data, understand, judge. What I affirm as real IS what I have understood and judged. Positions invite development, not reversal.",
        "is_virtually_unconditioned": True
    },
]


# ============================================================================
# Level 4: Decision - Deliberation and Commitment
# ============================================================================
# Question: Is it good? Is it worth doing?
# The question differs from Level 3 in its OBJECT: not fact but value
# Judgment of value has same STRUCTURE as judgment of fact but different CONTENT

DECISION_TEMPLATES = [
    # MATHEMATICAL DOMAIN
    # Objects of deliberation: Which proof strategy? Which problem to pursue?
    # Values at stake: Elegance, simplicity, fruitfulness, generality
    {
        "domain": DataDomain.MATHEMATICAL,
        "facts": [
            "The problem can be solved by direct computation (tedious but certain)",
            "An indirect proof by contradiction might be more elegant",
            "The indirect approach, if successful, would reveal deeper structure"
        ],
        "values_at_stake": [
            "Certainty of reaching a solution",
            "Elegance and economy of proof",
            "Potential insight into broader mathematical structure"
        ],
        "options": [
            "Pursue direct computation",
            "Attempt proof by contradiction",
            "Try both in parallel"
        ],
        "evaluation": "Direct computation guarantees result but forgoes insight. Proof by contradiction risks failure but promises deeper understanding if successful. The mathematical value of elegance is not mere aesthetics - elegant proofs often reveal connections invisible to brute force.",
        "decision": "Attempt the indirect proof first, with direct computation as fallback if time permits.",
        "authenticity_check": "This aligns with being intelligent (seeking insight) while being reasonable (having a fallback). The decision is not arbitrary preference but grounded in what mathematics VALUES: understanding over mere calculation.",
        "responsibility_note": "I take responsibility for this choice. If the indirect approach fails and time runs out, the consequence is mine to bear."
    },

    # EMPIRICAL SCIENTIFIC DOMAIN
    # Objects of deliberation: Which experiment? Which hypothesis to pursue?
    # Values at stake: Truth, utility, ethical constraints
    {
        "domain": DataDomain.EMPIRICAL_SCIENTIFIC,
        "facts": [
            "Two competing hypotheses can explain the data",
            "Experiment A would test hypothesis 1 definitively but requires animal subjects",
            "Experiment B would test hypothesis 2 with human volunteers but yields probabilistic results",
            "Both experiments are within budget and capability"
        ],
        "values_at_stake": [
            "Definitiveness of scientific answer",
            "Ethical treatment of animal subjects",
            "Human volunteer safety and consent",
            "Efficient use of research resources"
        ],
        "options": [
            "Conduct Experiment A (animal study)",
            "Conduct Experiment B (human volunteer study)",
            "Design a third experiment that avoids both concerns"
        ],
        "evaluation": "Scientific value favors definitiveness, but ethical values constrain means. Experiment A's definitiveness comes at ethical cost that requires justification beyond scientific curiosity. Experiment B respects human autonomy but yields less certain results. A third design might sacrifice some definitiveness for ethical improvement.",
        "decision": "Explore a third experimental design first. If impossible, proceed with Experiment B (human volunteers with proper consent), accepting probabilistic results as ethically preferable.",
        "authenticity_check": "Being responsible means not treating any value as absolute. Scientific truth is a value but not the ONLY value. This decision integrates multiple legitimate values rather than optimizing one at all costs.",
        "responsibility_note": "I acknowledge that accepting probabilistic results may delay scientific progress. I take responsibility for this trade-off as an authentic balancing of values."
    },

    # COMMON SENSE DOMAIN
    # Objects of deliberation: What to do in this situation?
    # Values at stake: Practical goods, relationships, welfare
    {
        "domain": DataDomain.COMMON_SENSE,
        "facts": [
            "Meeting is at 3:00 PM, travel needs 30+ minutes with traffic",
            "Report due at 2:00 PM is not yet started",
            "Boss strongly values punctuality",
            "I haven't eaten since last night",
            "Lunch hour is the only work window"
        ],
        "values_at_stake": [
            "Professional reliability (meeting attendance, boss's trust)",
            "Work quality (report completion)",
            "Physical wellbeing (need to eat)",
            "Personal integrity (not submitting substandard work)"
        ],
        "options": [
            "Skip lunch, complete report, risk energy crash at meeting",
            "Eat lunch, incomplete report, arrive at meeting prepared",
            "Communicate proactively: inform boss of situation, negotiate"
        ],
        "evaluation": "All options involve trade-offs. Heroic effort (skip lunch, complete report, make meeting) risks failure on all fronts if energy crashes. Prioritizing one value (punctuality) over others (report quality, health) is not automatically correct. Proactive communication treats the boss as an intelligent collaborator rather than an obstacle.",
        "decision": "Email boss now explaining the conflict, propose either (a) brief meeting delay, or (b) submitting report draft by 2PM with final by EOD. Eat something quick. This addresses all values honestly.",
        "authenticity_check": "Being responsible means owning the situation, not hiding from it. Proactive communication is more authentic than silent heroism that might fail. It treats relationships as genuine rather than purely transactional.",
        "responsibility_note": "I take responsibility for how this reflects on me. The boss may be displeased regardless. But honest communication maintains integrity even if it has costs."
    },

    # POP SCIENCE / COMMON NONSENSE DOMAIN
    # CRITICAL: Here deliberation is CORRUPTED by false judgments
    # Training model to recognize how counterpositions corrupt decision
    {
        "domain": DataDomain.POP_SCIENCE,
        "facts": [
            "A viral article claims new supplement 'boosts immunity' (based on one small study, not replicated)",
            "Many friends are taking it and report 'feeling better'",
            "The supplement is expensive but 'what's money compared to health?'",
            "No mainstream medical organization recommends it yet"
        ],
        "values_at_stake": [
            "Health (genuine value)",
            "Not being left behind socially",
            "Prudent use of money",
            "Authentic self-care vs. anxiety-driven consumption"
        ],
        "options": [
            "Buy and take the supplement like friends",
            "Wait for better evidence before deciding",
            "Consult a doctor for personalized advice"
        ],
        "evaluation": "THIS DELIBERATION IS CORRUPTED AT ITS BASE. The 'facts' include a pseudo-judgment (the supplement 'boosts immunity' based on inadequate evidence). Decisions built on false judgments are unreliable regardless of how carefully we deliberate. 'Feeling better' is anecdotal. Social belonging is irrelevant to efficacy. The first step is recognizing the epistemic problem, not choosing between options.",
        "decision": "REFUSE the premises of the deliberation. The 'fact' that the supplement works is not established. Return to Level 3: What is the actual evidence? Until that judgment is made responsibly, 'deciding' to take or not take is premature. Consult reliable sources, not social proof.",
        "authenticity_check": "Being responsible means not acting on pseudo-knowledge. The pressure to decide NOW is itself part of the manipulation (fear of missing out, social proof). Authentic responsibility sometimes means STOPPING to verify before acting.",
        "responsibility_note": "I take responsibility for the delay. If the supplement turns out to be effective and I missed early benefits, that's my cost. But acting on insufficient evidence is not authentic responsibility - it's anxiety pretending to be decisiveness."
    },

    # PHILOSOPHIC DOMAIN
    # Objects of deliberation: What kind of knower am I becoming?
    # Values at stake: Intellectual, moral conversion; authenticity
    # This is the GROUND of all other deliberation
    {
        "domain": DataDomain.PHILOSOPHIC,
        "facts": [
            "I have discovered the invariant structure of my own knowing: attend, understand, judge, decide",
            "I recognize that counterpositions (treating knowing as 'looking') are self-defeating",
            "I face a choice: live according to this self-appropriation, or continue in pre-reflective habits",
            "Authentic living requires ongoing attentiveness, intelligence, reasonableness, responsibility"
        ],
        "values_at_stake": [
            "Intellectual authenticity (being a genuine knower)",
            "Moral authenticity (acting on what I truly know)",
            "Personal integrity (consistency between insight and action)",
            "The value of value itself (why does any of this matter?)"
        ],
        "options": [
            "Commit to the transcendental precepts as normative for my life",
            "Acknowledge intellectually but compartmentalize - continue with pre-reflective habits",
            "Reject the analysis and return to naive knowing"
        ],
        "evaluation": "Option 3 is self-defeating: to reject the analysis I must understand and judge it - performing the operations I reject. Option 2 is inauthenticity: knowing the truth but not living it. Option 1 is the only coherent path, but it demands ongoing conversion, not one-time decision.",
        "decision": "Commit to the ongoing project of authenticity: being attentive, intelligent, reasonable, responsible. This is not a rule imposed from outside but the immanent demand of my own conscious operations.",
        "authenticity_check": "This is the meta-level of authenticity. The question 'Am I being authentic?' can itself be asked inauthentically or authentically. The criterion is whether I am performing the operations genuinely - not merely saying the right words.",
        "responsibility_note": "This decision grounds all other responsibility. To be responsible in any particular domain requires being responsible in general - which is what this commitment is. I take responsibility for the project of my own existence as a knower."
    },
]


def generate_attention_sample(template: Dict[str, Any]) -> AttentionSample:
    """
    Generate an Attention (Level 1) sample from a template.

    Level 1 is "improve data through description" - attending to data AS data for inquiry,
    not as "already out there now."
    """
    return AttentionSample(
        raw_input=template["raw_input"],
        structured_output=template["structured_output"],
        data_type=template["data_type"],
        integrity_notes=template.get("integrity_notes")
    )


def generate_understanding_sample(template: Dict[str, Any]) -> UnderstandingSample:
    """Generate an Understanding (Level 2) sample from a template."""
    return UnderstandingSample(
        structured_input=template["structured_input"],
        question=template["question"],
        proposition=template["proposition"],
        reasoning=template["reasoning"],
        alternatives=template.get("alternatives", []),
        confidence_note="This is a hypothesis, not yet affirmed as true."
    )


def generate_judgment_sample(template: Dict[str, Any]) -> JudgmentSample:
    """Generate a Judgment (Level 3) sample from a template."""
    return JudgmentSample(
        proposition=template["proposition"],
        evidence=template["evidence"],
        conditions=template["conditions"],
        conditions_fulfilled=template["conditions_fulfilled"],
        judgment=template["judgment"],
        judgment_reasoning=template["judgment_reasoning"],
        is_virtually_unconditioned=template["is_virtually_unconditioned"]
    )


def generate_decision_sample(template: Dict[str, Any]) -> DecisionSample:
    """
    Generate a Decision (Level 4) sample from a template.

    Level 4 asks: "Is it good? Is it worth doing?"
    The question differs from Level 3 in its OBJECT: not fact but value.
    """
    return DecisionSample(
        facts=template["facts"],
        values_at_stake=template["values_at_stake"],
        options=template["options"],
        evaluation=template["evaluation"],
        decision=template["decision"],
        authenticity_check=template["authenticity_check"],
        responsibility_note=template["responsibility_note"]
    )


def generate_attention_dataset(
    num_samples: int,
    output_path: Optional[Path] = None,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Generate a dataset for Level 1 (Attention) training.

    The samples teach the model to:
    1. Receive raw, unstructured data
    2. "Improve data through description"
    3. Present data FOR inquiry (not as "already out there now")
    4. Preserve data integrity while adding structure
    """
    random.seed(seed)

    samples = []
    level_spec = get_level(CognitiveLevel.ATTENTION)

    for i in range(num_samples):
        template = ATTENTION_TEMPLATES[i % len(ATTENTION_TEMPLATES)].copy()

        sample = generate_attention_sample(template)
        training_format = sample_to_training_format(sample, CognitiveLevel.ATTENTION)
        chat_format = format_for_chat(training_format)

        samples.append({
            "messages": chat_format,
            "metadata": {
                "level": "attention",
                "domain": template["domain"].value,
                "data_type": template["data_type"],
                "sample_id": i
            }
        })

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
        print(f"Saved {len(samples)} Attention samples to {output_path}")

    return samples


def generate_understanding_dataset(
    num_samples: int,
    output_path: Optional[Path] = None,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Generate a dataset for Level 2 (Understanding) training.

    The samples teach the model to:
    1. Receive structured data
    2. Ask "What is it? Why is it so?"
    3. Generate propositions that unify the data
    4. NOT affirm truth - only offer hypotheses
    """
    random.seed(seed)

    samples = []
    level_spec = get_level(CognitiveLevel.UNDERSTANDING)

    for i in range(num_samples):
        # Cycle through templates with variation
        template = UNDERSTANDING_TEMPLATES[i % len(UNDERSTANDING_TEMPLATES)].copy()

        # Create sample
        sample = generate_understanding_sample(template)

        # Convert to training format
        training_format = sample_to_training_format(sample, CognitiveLevel.UNDERSTANDING)
        chat_format = format_for_chat(training_format)

        samples.append({
            "messages": chat_format,
            "metadata": {
                "level": "understanding",
                "domain": template["domain"].value,
                "sample_id": i
            }
        })

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
        print(f"Saved {len(samples)} Understanding samples to {output_path}")

    return samples


def generate_judgment_dataset(
    num_samples: int,
    output_path: Optional[Path] = None,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Generate a dataset for Level 3 (Judgment) training.

    The samples teach the model to:
    1. Receive a proposition and evidence
    2. Identify the conditions for the proposition's truth
    3. Check fulfillment of conditions in evidence
    4. Affirm (Yes), Deny (No), or Withhold based on virtually unconditioned
    """
    random.seed(seed)

    samples = []
    level_spec = get_level(CognitiveLevel.JUDGMENT)

    for i in range(num_samples):
        # Cycle through templates
        template = JUDGMENT_TEMPLATES[i % len(JUDGMENT_TEMPLATES)].copy()

        # Create sample
        sample = generate_judgment_sample(template)

        # Convert to training format
        training_format = sample_to_training_format(sample, CognitiveLevel.JUDGMENT)
        chat_format = format_for_chat(training_format)

        samples.append({
            "messages": chat_format,
            "metadata": {
                "level": "judgment",
                "domain": template["domain"].value,
                "judgment_type": template["judgment"],
                "is_virtually_unconditioned": template["is_virtually_unconditioned"],
                "sample_id": i
            }
        })

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
        print(f"Saved {len(samples)} Judgment samples to {output_path}")

    return samples


def generate_decision_dataset(
    num_samples: int,
    output_path: Optional[Path] = None,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Generate a dataset for Level 4 (Decision) training.

    The samples teach the model to:
    1. Receive judgments of fact and values at stake
    2. Deliberate across options
    3. Make value judgments grounded in authenticity
    4. Take responsibility for the commitment
    """
    random.seed(seed)

    samples = []
    level_spec = get_level(CognitiveLevel.DECISION)

    for i in range(num_samples):
        template = DECISION_TEMPLATES[i % len(DECISION_TEMPLATES)].copy()

        sample = generate_decision_sample(template)
        training_format = sample_to_training_format(sample, CognitiveLevel.DECISION)
        chat_format = format_for_chat(training_format)

        samples.append({
            "messages": chat_format,
            "metadata": {
                "level": "decision",
                "domain": template["domain"].value,
                "sample_id": i
            }
        })

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
        print(f"Saved {len(samples)} Decision samples to {output_path}")

    return samples


def generate_full_pipeline_sample(
    situation: str,
    attention_output: str,
    understanding_template: Dict[str, Any],
    judgment_template: Dict[str, Any],
    decision_template: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate a sample showing all four levels processing the same situation.

    This demonstrates:
    1. How levels build on each other
    2. How semantic meaning shifts across levels
    3. The cumulative nature of cognitional process
    """
    attention = AttentionSample(
        raw_input=situation,
        structured_output=attention_output,
        data_type="mixed"
    )

    understanding = generate_understanding_sample(understanding_template)
    judgment = generate_judgment_sample(judgment_template)

    return {
        "situation": situation,
        "attention": asdict(attention),
        "understanding": asdict(understanding),
        "judgment": asdict(judgment),
        "decision": asdict(DecisionSample(**decision_template)) if decision_template else None
    }


def validate_judgment_sample(sample: JudgmentSample) -> Tuple[bool, List[str]]:
    """
    Validate that a Judgment sample correctly implements the virtually unconditioned.

    Returns (is_valid, list of issues).

    Note on denial: There are TWO valid forms of "No" judgment:
    1. Conditions not fulfilled -> uncertain denial (is_virtually_unconditioned = False)
    2. Conditions of PROOF fulfilled -> certain denial of proposition as IMPOSSIBLE
       (is_virtually_unconditioned = True)

    In mathematical proofs by contradiction, all conditions being fulfilled proves
    the proposition impossible. This is a VALID denial with certainty.
    """
    issues = []

    # Check conditions match fulfillment keys
    if set(sample.conditions) != set(sample.conditions_fulfilled.keys()):
        issues.append("Conditions list doesn't match fulfillment dictionary keys")

    # Check judgment aligns with conditions
    all_fulfilled = all(sample.conditions_fulfilled.values())
    any_false = any(not v for v in sample.conditions_fulfilled.values())

    if sample.judgment == "Yes" and not all_fulfilled:
        issues.append("Judgment is 'Yes' but not all conditions are fulfilled")

    # "No" with all conditions fulfilled is valid IF is_virtually_unconditioned is True
    # This represents proof of impossibility (e.g., Euclid's proof no largest prime exists)
    if sample.judgment == "No" and all_fulfilled and not sample.is_virtually_unconditioned:
        issues.append("Judgment is 'No' with conditions fulfilled but is_virtually_unconditioned is False - should be True for certain denial")

    if sample.judgment == "Yes" and not sample.is_virtually_unconditioned:
        issues.append("Judgment is 'Yes' but is_virtually_unconditioned is False")

    # "Insufficient evidence" should always have is_virtually_unconditioned = False
    if "Insufficient" in sample.judgment and sample.is_virtually_unconditioned:
        issues.append("Judgment withholds but is_virtually_unconditioned is True - should be False when evidence insufficient")

    return len(issues) == 0, issues


if __name__ == "__main__":
    from pathlib import Path

    # Generate sample datasets for all four levels
    data_dir = Path(__file__).parent.parent / "data" / "synthetic"

    print("=" * 60)
    print("GENERATING TRAINING DATA FOR FOUR COGNITIVE LEVELS")
    print("Five domains: Mathematical, Empirical, Common Sense, Pop Science, Philosophic")
    print("=" * 60)

    print("\n[Level 1] Generating Attention dataset...")
    print("  (Improve data through description - present data FOR inquiry)")
    attention_samples = generate_attention_dataset(
        num_samples=10,
        output_path=data_dir / "attention_samples.jsonl"
    )

    print("\n[Level 2] Generating Understanding dataset...")
    print("  (What is it? Why? - generate hypotheses, NOT truth claims)")
    understanding_samples = generate_understanding_dataset(
        num_samples=10,
        output_path=data_dir / "understanding_samples.jsonl"
    )

    print("\n[Level 3] Generating Judgment dataset...")
    print("  (Is it so? - grasp virtually unconditioned, affirm/deny/withhold)")
    judgment_samples = generate_judgment_dataset(
        num_samples=10,
        output_path=data_dir / "judgment_samples.jsonl"
    )

    print("\n[Level 4] Generating Decision dataset...")
    print("  (Is it good? - deliberate on values, commit responsibly)")
    decision_samples = generate_decision_dataset(
        num_samples=10,
        output_path=data_dir / "decision_samples.jsonl"
    )

    # Validate judgment samples (most critical for counterposition detection)
    print("\n" + "=" * 60)
    print("VALIDATING JUDGMENT SAMPLES (Level 3)")
    print("Checking virtually unconditioned structure...")
    print("=" * 60)
    for template in JUDGMENT_TEMPLATES:
        sample = generate_judgment_sample(template)
        is_valid, issues = validate_judgment_sample(sample)
        status = "✓" if is_valid else "✗"
        domain = template["domain"].value
        print(f"  {status} [{domain:15}] {template['proposition'][:45]}...")
        for issue in issues:
            print(f"      Issue: {issue}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Attention templates:    {len(ATTENTION_TEMPLATES)} ({len(set(t['domain'] for t in ATTENTION_TEMPLATES))} domains)")
    print(f"  Understanding templates: {len(UNDERSTANDING_TEMPLATES)} ({len(set(t['domain'] for t in UNDERSTANDING_TEMPLATES))} domains)")
    print(f"  Judgment templates:     {len(JUDGMENT_TEMPLATES)} ({len(set(t['domain'] for t in JUDGMENT_TEMPLATES))} domains)")
    print(f"  Decision templates:     {len(DECISION_TEMPLATES)} ({len(set(t['domain'] for t in DECISION_TEMPLATES))} domains)")
