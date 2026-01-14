# Function Calling at Scale: How Performance Degrades with Large Tool Libraries

## Abstract

We investigate how large language model (LLM) function-calling performance scales with the number of available tools. Using the Berkeley Function Calling Leaderboard (BFCL) framework, we systematically evaluate five state-of-the-art models (Grok-4, Grok-3-beta, Grok-3-mini-beta, GPT-4o, GPT-4o-mini) across function pool sizes ranging from baseline (1-4 functions) to 512 functions. Our findings reveal consistent degradation patterns across all models, with smaller models showing significantly steeper decline. At 512 functions, top-tier models (Grok-4, GPT-4o, Grok-3-beta) maintain 63-66% accuracy compared to 70-74% baseline, while smaller models drop to 45-48%. Analysis of failure modes shows parameter accuracy, not function retrieval, drives the degradation. Context window limits at 1024 functions present a practical constraint for current architectures.

## 1. Introduction

Function calling enables LLMs to interact with external tools and APIs, but real-world applications often provide access to hundreds or thousands of functions. While existing benchmarks like BFCL evaluate models with small tool sets (<10 functions), it remains unclear how performance scales to realistic library sizes.

We address this gap by systematically scaling function pool sizes from 1 to 1024 functions and measuring degradation patterns across model tiers. Our key findings:

1. **Non-linear degradation**: Performance remains stable up to ~64 functions, then accelerates
2. **Model tier matters**: Larger models show 2x better robustness to scaling
3. **Parameter precision, not retrieval**: 85% of failures involve correct function selection but wrong/missing parameters
4. **Context limits**: All tested models hit context window limits at 1024 functions

## 2. Methodology

### 2.1 Test Set Construction

Starting from BFCL's `simple_python` category (400 test cases, single function calls), we:

1. Extracted 2,405 unique functions across all BFCL categories
2. For each test, injected N distractor functions from this pool
3. Shuffled function order to avoid positional bias
4. Generated test sets for pool sizes: 0, 16, 32, 64, 128, 256, 512, 1024

Each test maintains its original question and ground truth function, surrounded by semantically diverse distractors.

### 2.2 Models Evaluated

| Model | Context Window | Cost Tier |
|-------|---------------|-----------|
| Grok-4 | ~128K | High |
| Grok-3-beta | 131K | High |
| GPT-4o | ~128K | High |
| GPT-4o-mini | ~128K | Low |
| Grok-3-mini-beta | 131K | Low |

### 2.3 Evaluation Protocol

We use an LLM judge (GPT-4o) to assess semantic equivalence, accounting for:
- Functionally equivalent function names (e.g., `calculate_area` vs `compute_area`)
- Parameter name variations with same meaning
- Different argument representations for equivalent values

This addresses limitations of exact-match evaluation in scenarios where multiple valid implementations exist.

## 3. Results

### 3.1 Overall Performance by Pool Size

| Pool Size | Grok-4 | Grok-3-beta | GPT-4o | GPT-4o-mini | Grok-3-mini-beta |
|-----------|---------|-------------|---------|-------------|------------------|
| 0 (baseline) | 74.00% | **73.00%** | 74.00% | 70.25% | 66.00% |
| 16 | 72.00% | 70.00% | 72.50% | 67.00% | 62.50% |
| 32 | 71.50% | 69.25% | 71.75% | 66.25% | 60.75% |
| 64 | 70.50% | 68.00% | 70.00% | 63.25% | 56.00% |
| 128 | 69.00% | 66.50% | 68.50% | 60.25% | 52.50% |
| 256 | 67.25% | 64.50% | 65.75% | 55.50% | 51.00% |
| 512 | **65.50%** | 62.75% | 63.00% | 48.00% | 45.00% |
| 1024 | Context limit | Context limit | Context limit | Context limit | Context limit |

**Key observations:**
- Grok-4 leads at 512 functions (65.50%), followed by GPT-4o (63.00%) and Grok-3-beta (62.75%)
- Grok-3-beta achieves highest baseline accuracy (73.00%) and maintains competitive performance throughout
- Mini models show 2x steeper degradation (-21% to -22% vs -10% to -11% for top-tier)
- All models hit context limits at 1024 functions (~128-131K token windows)

### 3.2 Degradation Patterns

**Total degradation from baseline to pool 512:**
- Grok-4: -8.50% (most robust)
- Grok-3-beta: -10.25%
- GPT-4o: -11.00%
- GPT-4o-mini: -22.25%
- Grok-3-mini-beta: -21.00% (steepest decline)

**Critical degradation regions:**
- Pools 0-64: Gradual decline (~1% per doubling)
- Pools 64-256: Accelerating decline (~3-4% per doubling)
- Pools 256-512: Severe decline (~5-8% per doubling)

The non-linear pattern suggests a cognitive threshold around 64-128 functions where models transition from manageable to overwhelming function sets.

### 3.3 Failure Mode Analysis

We analyzed 208 failures in GPT-4o-mini at pool 512 where baseline succeeded:

**Failure breakdown:**
- **Missing parameters**: 41% - Correct function, but omitted required/optional parameters
- **Wrong parameter values**: 44% - Correct function, but incorrect argument values or malformed data
- **Wrong function selection**: 12% - Similar function that can't fulfill requirements
- **Completely unrelated function**: 3%
- **Hallucinated functions**: 0%

**Key insight**: 97% of failures involve correct or similar function selection. The primary issue is parameter accuracy, not retrieval.

### 3.4 Concrete Failure Examples

**Example 1: Complete breakdown**
- Question: "Calculate hypotenuse with sides 4 and 5"
- Baseline: `math.hypot(x=4, y=5)` ✓
- Pool 512: `multiply(a=4, b="25)", multiply(a=5, add(a=16, math.hypot(x=4, y=5))` ✗
- Pattern: Model output malformed, mixed multiple function names into arguments

**Example 2: Function downgrade**
- Question: "Circumference of circle with radius 4 inches"
- Baseline: `calculate_circumference(radius=4, unit="inches")` ✓
- Pool 512: `circle_properties.get(radius=4)` ✗
- Pattern: Switched to generic function, lost specificity and unit parameter

**Example 3: Parameter omission**
- Question: "Calculate final velocity of object falling from 150m building"
- Baseline: `calculate_final_velocity(height=150, initial_velocity=0, gravity=9.81)` ✓
- Pool 512: `calculate_final_velocity(height=150, initial_velocity=0)` ✗
- Pattern: Dropped optional `gravity` parameter that affects result

## 4. Discussion

### 4.1 Why Parameter Precision Degrades

The dominance of parameter errors (85%) over retrieval errors (3%) suggests models successfully navigate large function sets but struggle with parameter fidelity. Possible mechanisms:

1. **Attention dilution**: With 512+ functions in context, model attention spreads across more tokens, reducing focus on parameter specifications
2. **Default assumption bias**: Models assume optional parameters have "reasonable" defaults rather than inferring specific values
3. **Complex parameter handling**: List/range parameters are more likely to be truncated or malformed (e.g., `interval=[1,3]` becomes `interval="[1"`)

### 4.2 Model Robustness Factors

Top-tier models (Grok-4, GPT-4o, Grok-3-beta) maintain 2x better accuracy than mini models at scale. Potential factors:

1. **Model capacity**: Larger models may have more robust function-calling training
2. **Instruction following**: Better instruction adherence preserves parameter requirements
3. **Context utilization**: More effective use of long-context information

Notably, Grok-3-beta achieves highest baseline and stays competitive despite being a smaller model than Grok-4, suggesting architecture and training may matter more than raw size.

### 4.3 Practical Implications

**For developers:**
- Consider function filtering/ranking for large tool libraries
- Test critical paths with realistic function pool sizes
- Implement validation for missing/incorrect parameters

**For model providers:**
- Context window size alone insufficient for 1000+ function scenarios
- Parameter precision at scale needs targeted improvement
- Function calling evaluation should include realistic scale tests

### 4.4 Limitations

1. **Single test category**: We evaluated only `simple_python` (single function calls); multi-turn and parallel calling may show different patterns
2. **Distractor diversity**: Random sampling may not represent realistic function similarity distributions
3. **Judge limitations**: LLM judge evaluation, while flexible, may have its own biases
4. **Context window**: We couldn't test beyond 512 functions effectively due to model limits

## 5. Conclusions

Function-calling performance degrades consistently but non-uniformly as tool libraries scale. The 64-function threshold marks a transition from graceful to severe degradation, with parameter precision (not function retrieval) driving failures. Top-tier models show 2x better robustness but still lose 10-11% accuracy by 512 functions. Current context windows limit practical evaluation beyond this scale.

Future work should explore:
- Function ranking/filtering strategies to maintain high precision
- Targeted parameter validation mechanisms
- Multi-turn function calling at scale
- Architectural improvements for parameter fidelity

The gap between mini and full-size models suggests this remains an active area for improvement, with significant implications for real-world agent deployment.

## Data and Code

All experimental data, evaluation scripts, and analysis code available at:
https://github.com/itea1001/bfcl-thousand-tools-01

