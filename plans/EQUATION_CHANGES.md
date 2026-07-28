# Proposal: Improving the ESG Formula Estimator

## Background

Our current deterministic ESG estimator uses the following equation:

\[
Score = B + \sum_i w_i c_i \delta_i
\]

Where:

- **B** = Country baseline ESG score
- **wᵢ** = Weight assigned to the ESG factor
- **cᵢ** = Confidence of the extracted claim
- **δᵢ** = Factor impact (between -1 and +1)

This is a good starting point because it is:

- Deterministic
- Explainable
- Fast
- Easy to audit
- Easy to calibrate

However, mathematically it assumes that every factor behaves **linearly**, which is not how ESG evidence behaves in reality.

---

# Problems with the Current Formula

## 1. Linear contribution

Every additional piece of evidence changes the score by exactly the same amount.

Example:

Environmental Controversies

Current model:

```
1 controversy  -> -10

2 controversies -> -20

5 controversies -> -50
```

This is unrealistic.

In reality,

- the first controversy damages reputation significantly,
- the second adds more damage,
- but the twentieth controversy does **not** make the company twenty times worse.

The impact naturally saturates.

---

## 2. Confidence scales linearly

Current:

```
Confidence = 1.0 -> 100%

Confidence = 0.8 -> 80%

Confidence = 0.5 -> 50%
```

This means mediocre evidence still contributes heavily.

In practice,

high-confidence evidence should dominate,
while uncertain evidence should contribute much less.

---

## 3. Positive metrics also saturate

Example:

Renewable Energy %

Current model:

```
20% -> +2

40% -> +4

80% -> +8
```

Reality:

Moving from

20% → 40%

is a much bigger improvement than

80% → 100%.

Again,

linear equations cannot represent diminishing returns.

---

# Why Calculus?

My guide suggested looking into calculus instead of pure algebra.

The purpose is **not** because integration itself magically improves accuracy.

Rather,

calculus provides continuous functions that naturally model:

- diminishing returns
- saturation
- continuous evidence accumulation
- smooth score changes

These properties fit ESG scoring much better than a simple weighted sum.

---

# Proposed Mathematical Improvement

Instead of

\[
Score = B + \sum_i w_i c_i \delta_i
\]

we model ESG as the accumulation of evidence over a continuous evidence space:

\[
Score =
B +
\int_0^N
w(x)c(x)g(\delta(x))
dx
\]

Where

- **w(x)** = importance of evidence
- **c(x)** = confidence
- **g(δ)** = nonlinear impact function

The integral simply represents accumulating evidence continuously instead of assuming each contribution is perfectly linear.

---

# Practical Implementation

Our pipeline does **not** have continuous data.

We have discrete claims.

Therefore the integral naturally becomes a numerical summation:

\[
Score
=
B
+
\sum_i
w_i
c_i
g(\delta_i)
\]

This is mathematically equivalent to approximating the integral using discrete observations.

So implementation complexity remains almost identical to today.

---

# Choosing the Impact Function

Instead of

```
g(δ)=δ
```

I propose

\[
g(\delta)=\tanh(k\delta)
\]

where

k controls how quickly the curve saturates.

Example (k = 1)

| δ | Linear | tanh(δ) |
|---|--------|----------|
|0.2|0.20|0.20|
|0.5|0.50|0.46|
|1.0|1.00|0.76|
|2.0|2.00|0.96|
|5.0|5.00|0.999|

Notice that

large impacts gradually flatten instead of increasing forever.

This matches real ESG behaviour much better.

---

# Confidence Improvement

Instead of

\[
Contribution
=
w_i
c_i
g(\delta_i)
\]

I would use

\[
Contribution
=
w_i
c_i^2
g(\delta_i)
\]

Squaring confidence gives much more influence to highly reliable evidence while naturally suppressing uncertain evidence.

Example

| Confidence | Linear | Squared |
|------------|--------|----------|
|1.0|1.00|1.00|
|0.9|0.90|0.81|
|0.8|0.80|0.64|
|0.6|0.60|0.36|
|0.4|0.40|0.16|

This better reflects real-world trust.

---

# Architectural Improvement

I believe an even bigger improvement than changing the equation is changing what each extracted claim represents.

Currently each claim becomes

```
One Claim
↓

One Contribution
```

Instead,

each claim should become a richer probabilistic observation.

Each claim should carry:

- extraction confidence
- source reliability
- recency
- evidence strength
- sector relevance
- contradiction count

instead of only confidence and delta.

---

# Evidence Potential

Define an evidence potential

\[
\Psi_i
=
r_i
c_i
e^{-\lambda\Delta t_i}
\tanh(k\delta_i)
\]

Where

- **rᵢ** = source reliability
- **cᵢ** = extraction confidence
- **e^{-λΔt}** = time decay
- **tanh(kδ)** = bounded nonlinear impact

This converts every claim into a standardized "evidence contribution".

---

# Final ESG Equation

The final score becomes

\[
Score
=
B
+
\sum_i
w_i
\Psi_i
\]

or

\[
Score
=
B
+
\sum_i
w_i
\left(
r_i
c_i
e^{-\lambda\Delta t_i}
\tanh(k\delta_i)
\right)
\]

This is still:

- deterministic
- explainable
- easy to audit
- easy to calibrate

while incorporating

- calculus (continuous decay and nonlinear functions),
- statistics (confidence and reliability),
- bounded contributions,
- diminishing returns.

---

# Comparison

| Current Formula | Proposed Formula |
|----------------|------------------|
|Linear contributions|Nonlinear bounded contributions|
|Unlimited impact|Saturating impact|
|Only confidence|Confidence + reliability + recency|
|No ageing of evidence|Old evidence gradually loses influence|
|Simple weighted sum|Continuous evidence accumulation|
|Can over-penalize repeated events|Naturally bounded influence|
|Manual weights only|Supports future learned weights|

---

# Computational Cost

Current:

```
O(number of claims)
```

Proposed:

```
O(number of claims)
```

The computational complexity does **not** change.

The only additional operations are

- tanh()
- exponential decay
- multiplication by reliability

which are negligible even for ~100,000 companies.

---

# Final Recommendation

I **would not** redesign the overall pipeline.

The existing architecture is already well-designed:

```
Signals
    ↓
Claims
    ↓
Formula Estimator
    ↓
Holistic Estimator
    ↓
Reconciliation
    ↓
Verification
```

The change should be isolated to the Formula Estimator.

Specifically:

1. Replace the purely linear contribution with a nonlinear bounded function (`tanh`).
2. Introduce evidence reliability and time decay into each claim.
3. Continue using deterministic weighted aggregation.
4. Keep the current architecture unchanged.
5. Later, use calibration data (BCorp/Upright) to learn optimal weights instead of hand-tuning them.

This preserves all of the strengths of the current deterministic pipeline while making the mathematics significantly more representative of how ESG evidence behaves in practice.

---

## CHANGES 2.0

# Final Recommendation for Improving the ESG Formula Estimator

## Objective

Improve the mathematical behavior of the deterministic Formula Estimator **without changing the overall architecture**.

The existing pipeline is already well-designed:

```
Signals
    ↓
Evidence Extraction
    ↓
Claims
    ↓
Formula Estimator
    ↓
Holistic Estimator
    ↓
Reconciliation
    ↓
Verification
```

The recommendation is to improve **only the Formula Estimator**.

---

# Current Formula

The current estimator computes:

\[
Score = B + \sum_i w_i c_i \delta_i
\]

Where:

- **B** = Country baseline
- **wᵢ** = Factor weight
- **cᵢ** = Claim confidence (already includes freshness adjustments)
- **δᵢ** = Factor contribution (-1 to +1)

---

# Current Strengths

The existing formula has several advantages:

- Fully deterministic
- Easy to audit
- Easy to explain
- Computationally inexpensive
- Easy to calibrate
- Produces per-factor contribution traces

These strengths should be preserved.

---

# What Should NOT Be Changed

After reviewing the current implementation, several previously proposed ideas are **not recommended**.

## 1. Do NOT add another time-decay term

The current implementation already applies exponential freshness decay through `evidence_freshness.py`.

Adding another

\[
e^{-\lambda\Delta t}
\]

inside the scoring equation would decay evidence twice.

Freshness should continue to remain part of the existing confidence calculation.

---

## 2. Do NOT square confidence

Earlier, the idea was

\[
Contribution = w_i c_i^2 \delta_i
\]

This is **not recommended** because:

- LLM confidence is not yet calibrated.
- Squaring confidence suppresses medium-confidence claims aggressively.
- There is currently no empirical evidence that this improves prediction accuracy.

Confidence should remain linear until calibration experiments prove otherwise.

---

## 3. Do NOT apply `tanh()` to every individual claim

Earlier proposal:

\[
Contribution = w_i c_i \tanh(k\delta_i)
\]

However, the current Formula Estimator already selects **only the single best claim per factor**.

Repeated news articles about the same controversy are already prevented from stacking.

Therefore, per-claim saturation solves a problem that has largely already been addressed.

---

# The Remaining Mathematical Gap

The current implementation still allows multiple **different factors** to accumulate linearly.

Example:

```
Environmental controversy   -10
Regulatory fines            -9
Litigation                  -7
Board governance            -8
--------------------------------
Total swing                -34
```

Each factor is valid individually, but together they may produce an unrealistically large movement away from the baseline.

This is the remaining limitation of the current linear model.

---

# Proposed Improvement

Instead of saturating each claim independently, compute the total evidence swing first.

## Step 1

Compute the weighted evidence swing exactly as today:

\[
\Delta =
\sum_i
w_i c_i \delta_i
\]

---

## Step 2

Normalize the swing by the total available factor weight:

\[
\Delta_{norm}
=
\frac
{\sum_i w_i c_i \delta_i}
{\sum_i w_i}
\]

This keeps the evidence swing within a stable range regardless of how many ESG factors exist.

Benefits:

- Adding more ESG factors does not inflate scores.
- Scores remain comparable across future versions.
- Weight interpretation stays consistent.
- Calibration becomes easier.

---

## Step 3

Apply one nonlinear saturation function to the **aggregate** evidence swing.

\[
Score
=
B
+
A
\tanh
(k\Delta_{norm})
\]

Where:

- **A** = Maximum allowable swing from the baseline
- **k** = Saturation parameter (controls how quickly the curve flattens)

This limits the total movement away from the baseline while still preserving the influence of all contributing factors.

---

# Why This Is Better

Current model:

```
Baseline

↓

+ Sum of all contributions

↓

Final Score
```

Every additional factor increases the score linearly forever.

---

Proposed model:

```
Baseline

↓

Total weighted evidence

↓

Normalize

↓

Smooth saturation (tanh)

↓

Final Score
```

Large amounts of supporting or opposing evidence gradually flatten instead of pushing the score indefinitely.

This better reflects real ESG assessments, where additional evidence eventually provides diminishing additional impact.

---

# Choosing the Saturation Parameter (k)

The value of **k** should **not** be hardcoded.

Instead, treat it as a tunable hyperparameter.

Example:

```
k = 0.25

↓

Calibration Harness

↓

Spearman Correlation

↓

k = 0.50

↓

Calibration Harness

↓

...

↓

Choose the best-performing value.
```

This follows the same philosophy already planned for learning factor weights from calibration data.

---

# Computational Complexity

Current:

```
O(number of claims)
```

Proposed:

```
O(number of claims)
```

The only additional operation is evaluating one `tanh()` function per pillar.

The computational cost remains effectively unchanged, making the approach suitable for scaling to 100,000+ companies.

---

# Final Proposed Formula

Current:

\[
Score
=
B
+
\sum_i
w_i c_i \delta_i
\]

Recommended:

\[
\Delta
=
\frac
{\sum_i w_i c_i \delta_i}
{\sum_i w_i}
\]

\[
Score
=
B
+
A
\tanh(k\Delta)
\]

where:

- **B** = Country baseline
- **wᵢ** = Factor weight
- **cᵢ** = Existing calibrated confidence (already includes freshness)
- **Δ** = Normalized weighted evidence swing
- **k** = Tunable saturation parameter
- **A** = Maximum allowable deviation from the baseline

---

# Summary

This recommendation intentionally makes the **smallest possible mathematical change** while preserving the existing architecture.

It:

- Keeps the deterministic scoring framework.
- Preserves explainability and auditability.
- Reuses the existing confidence and freshness logic.
- Prevents excessive score swings caused by multiple high-impact factors.
- Introduces a mathematically justified nonlinear saturation function.
- Maintains the same computational complexity.
- Can be validated objectively using the existing calibration harness before deployment.

Rather than redesigning the estimator, this approach refines the current weighted-sum model into a bounded, smoother scoring function that better reflects how ESG evidence accumulates in practice.

---
## CHANGES 3.0
# Final Mathematical Recommendation (v3)

## Objective

Improve the deterministic Formula Estimator while preserving the existing pipeline architecture, explainability, auditability, and computational efficiency.

The proposed changes should be isolated to the Formula Estimator and should not require modifications to the extraction, reconciliation, or verification stages.

---

# Existing Formula

The current estimator computes:

\[
Score = B + \sum_i w_i c_i \delta_i
\]

Where:

- **B** = Country baseline score
- **wᵢ** = Factor weight
- **cᵢ** = Confidence (already incorporates evidence freshness)
- **δᵢ** = Factor impact (-1 to +1)

This formulation is simple, deterministic, and easy to audit, but it assumes that evidence accumulates linearly.

---

# Current Strengths

The existing implementation already solves several important problems:

- Picks the **best claim per factor**, preventing repeated news articles from stacking.
- Applies **freshness decay** through `evidence_freshness.py`.
- Produces fully explainable per-factor contributions.
- Runs in linear time.
- Integrates cleanly with the current Formula → Holistic → Reconcile pipeline.

These behaviors should remain unchanged.

---

# Remaining Limitation

Although repeated claims for the same factor no longer stack, **multiple different high-impact factors still combine linearly**.

Example:

```
Environmental Controversy   -10
Regulatory Fine             -9
Litigation                  -8
Governance Failure          -7

Total Swing = -34
```

Every factor is individually valid, but together they can move the score unrealistically far from the country baseline.

The remaining problem is therefore **cross-factor accumulation**, not repeated-claim accumulation.

---

# Proposed Improvement

Instead of modifying each individual contribution, apply smoothing only after all evidence has been aggregated.

---

# Step 1 — Aggregate Evidence

For each pillar independently (Environmental, Social, Governance), compute the weighted evidence contribution exactly as today.

However, normalize it using **only the factors that actually contributed evidence**.

\[
\boxed{
\Delta
=
\frac
{\sum\limits_{i\in C}
w_i c_i \delta_i}
{\sum\limits_{i\in C}
w_i c_i}
}
\]

Where:

- **C** = set of factors that produced a valid claim
- **wᵢ** = factor weight
- **cᵢ** = existing confidence (already includes freshness adjustments)
- **δᵢ** = factor impact

---

# Why Normalize Using Only Contributing Factors?

The denominator should **not** be the total theoretical weight of every registered ESG factor.

Example:

```
Environmental Pillar

Total registry weight = 74

Company evidence:

Carbon Emissions
Weight = 10

Renewable Energy
Weight = 10
```

If divided by 74:

```
20 / 74 = 0.27
```

the company appears artificially weak simply because evidence is sparse.

Instead:

```
20 / 20 = 1.0
```

This correctly represents:

> "Among the evidence actually available, the company scores maximally."

This approach is much more consistent with sparse real-world ESG data.

---

# Properties of the Normalized Swing

The normalized evidence swing naturally satisfies

\[
-1 \le \Delta \le 1
\]

because:

- each factor contribution is bounded,
- confidence lies between 0 and 1,
- normalization uses only contributing evidence.

This immediately prevents uncontrolled score growth regardless of how many factors exist.

---

# Step 2 — Apply Smooth Saturation

After computing the normalized evidence swing, apply a nonlinear saturation function.

\[
\boxed{
Score
=
B
+
A_p
\tanh
(k_p\Delta)
}
\]

Where:

- **B** = Country baseline for the pillar
- **Aₚ** = Maximum deviation allowed for pillar *p*
- **kₚ** = Saturation parameter controlling curve steepness
- **Δ** = Normalized weighted evidence swing

---

# Why Keep tanh()?

Normalization already bounds the evidence between -1 and +1.

Therefore, `tanh()` is **not** primarily used to prevent runaway scores.

Instead, it provides **diminishing returns**.

Example:

| Normalized Swing | Linear | tanh() |
|-----------------|--------|---------|
|0.2|0.20|0.20|
|0.5|0.50|0.46|
|0.8|0.80|0.66|
|1.0|1.00|0.76|

This means:

- moderate evidence remains almost unchanged,
- increasingly strong evidence produces progressively smaller additional gains,
- extreme evidence does not dominate the final score.

This better reflects how ESG evidence behaves in practice.

---

# Parameter Selection

Neither **A** nor **k** should be hardcoded.

Both are calibration parameters.

### A (Maximum Deviation)

Controls how far a company may move from its country baseline.

Example:

```
Small A

↓

Tighter score range

Large A

↓

Wider score range
```

---

### k (Saturation Strength)

Controls how quickly diminishing returns begin.

Example:

```
Small k

↓

Almost linear

Large k

↓

Early saturation
```

---

# Calibration Strategy

Both parameters should be optimized using the existing calibration framework.

Example:

```
Candidate Parameters

↓

Calibration Harness

↓

Benchmark Comparison

↓

Spearman / Pearson Correlation

↓

Best Performing (A, k)
```

This follows the same philosophy already used for factor-weight calibration.

---

# Per-Pillar Calibration

The transformation should be applied **independently** to each ESG pillar.

Environmental, Social, and Governance have different:

- evidence density,
- factor distributions,
- benchmark characteristics.

Therefore each pillar may use different parameters:

```
Environmental

A_E
k_E

Social

A_S
k_S

Governance

A_G
k_G
```

This remains consistent with the existing reconciliation stage, which already treats pillars independently.

---

# Computational Complexity

Current implementation:

```
O(number of claims)
```

Proposed implementation:

```
O(number of claims)
```

Only a single normalization and one nonlinear function are added per pillar.

The computational cost remains effectively unchanged and easily scales to 100,000+ companies.

---

# Final Formula

For each ESG pillar:

\[
\boxed{
\Delta
=
\frac
{\sum\limits_{i\in C}
w_i c_i \delta_i}
{\sum\limits_{i\in C}
w_i c_i}
}
\]

\[
\boxed{
Score
=
B
+
A_p
\tanh(k_p\Delta)
}
\]

Where:

- **B** = Country baseline
- **C** = Contributing factors only
- **wᵢ** = Factor weight
- **cᵢ** = Existing confidence (already includes freshness)
- **δᵢ** = Factor contribution
- **Aₚ** = Maximum pillar deviation
- **kₚ** = Saturation parameter

---

# Summary

This proposal intentionally introduces the smallest possible mathematical change while preserving the existing architecture.

It:

- Preserves the deterministic weighted scoring framework.
- Reuses the current confidence and freshness mechanisms.
- Keeps per-factor explainability intact.
- Normalizes only over factors that actually contributed evidence.
- Addresses the remaining issue of cross-factor accumulation.
- Uses a smooth nonlinear transformation to model diminishing returns.
- Treats both **A** and **k** as tunable calibration parameters.
- Applies independently to Environmental, Social, and Governance pillars.
- Maintains the same computational complexity as the current implementation.

The result is a mathematically cleaner, bounded, and more realistic scoring model that integrates naturally with the existing Formula Estimator without requiring architectural changes.

---
## CHANGES 4.0
# Final ESG Formula Recommendation (v4)

## Objective

Improve the mathematical robustness of the deterministic Formula Estimator while preserving the existing architecture, explainability, and computational efficiency.

The guiding principles are:

- Keep the current pipeline unchanged.
- Make the smallest possible change to the Formula Estimator.
- Reuse existing mechanisms wherever possible.
- Introduce only mathematically meaningful improvements.
- Validate every tunable parameter using the existing calibration harness.

---

# Existing Pipeline

```
Signals
    ↓
Evidence Extraction
    ↓
Claims
    ↓
Formula Estimator
    ↓
Holistic Estimator
    ↓
Reconciliation
    ↓
Verification
```

No architectural changes are required.

---

# Existing Formula

The current implementation computes:

\[
Score = B + \sum_i w_i c_i \delta_i
\]

Where:

- **B** = Country baseline
- **wᵢ** = ESG factor weight
- **cᵢ** = Confidence (already incorporates evidence freshness)
- **δᵢ** = Factor impact (-1 to +1)

The current implementation already has several desirable properties:

- Best claim selected per factor
- Freshness decay already integrated
- Deterministic and explainable
- Linear computational complexity
- Easy contribution tracing

These strengths should remain unchanged.

---

# Remaining Limitation

The current implementation prevents repeated claims from stacking within the same factor.

However, multiple different high-impact factors still accumulate linearly.

Example:

```
Environmental Controversy   -10
Regulatory Fine             -9
Litigation                  -8
Governance Failure          -7

Total Swing = -34
```

Each individual contribution is reasonable.

The combined effect may become unrealistically large.

The remaining mathematical problem is therefore **cross-factor accumulation**, not repeated-claim accumulation.

---

# Proposed Formula

## Step 1 — Aggregate Evidence

For each ESG pillar independently:

\[
\boxed{
\Delta
=
\frac
{\sum\limits_{i\in C}
w_i c_i \delta_i}
{\sum\limits_{i\in C}
w_i}
}
\]

Where:

- **C** = only factors that contributed evidence
- **wᵢ** = factor weight
- **cᵢ** = confidence
- **δᵢ** = factor impact

This produces the average weighted ESG direction of the available evidence.

---

# Why Normalize Only Over Contributing Factors?

The denominator should **not** use the pillar's total theoretical weight.

Example:

```
Environmental registry

Total weight = 74

Company evidence:

Carbon Emissions
Weight = 10

Renewable Energy
Weight = 10
```

Dividing by 74 would incorrectly suppress companies with sparse evidence.

Instead,

```
20 / 20 = 1.0
```

correctly represents:

> "Among the evidence available for this company, the evidence is maximally positive."

---

# Step 2 — Coverage Adjustment

Normalization alone cannot distinguish:

```
Company A

12 well-supported ESG factors
```

from

```
Company B

1 well-supported ESG factor
```

Both could produce

```
Δ = 1.0
```

even though Company A has much stronger overall evidence.

Introduce a coverage metric:

\[
\boxed{
Coverage
=
\frac
{\sum\limits_{i\in C}w_i}
{\sum\limits_{i\in Pillar}w_i}
}
\]

Coverage naturally lies within

\[
0 \le Coverage \le 1
\]

---

# Coverage Multiplier

To avoid overly penalizing sparse but valid evidence, introduce a floor parameter β.

\[
\boxed{
CoverageMultiplier
=
\beta
+
(1-\beta)
\times
Coverage
}
\]

where

\[
0<\beta<1
\]

Example:

```
Coverage = 0.2

β = 0.6

Multiplier

=

0.6 + 0.4×0.2

=

0.68
```

This allows thin-but-real evidence to move the score while rewarding broader evidence coverage.

---

# Step 3 — Minimum Evidence Gate

A single weak claim should not significantly alter the baseline.

Define an evidence mass:

\[
EvidenceMass
=
\sum_{i\in C}
w_i c_i
\]

If

```
EvidenceMass < Threshold
```

then

```
Score = Baseline
```

This extends the existing invariant

```
No evidence

↓

Baseline
```

to

```
No meaningful evidence

↓

Baseline
```

---

# Step 4 — Smooth Saturation

Apply nonlinear saturation only after aggregation.

\[
\boxed{
Score
=
B
+
A_p
\tanh(k_p\Delta)
\times
CoverageMultiplier
}
\]

Where

- **B** = Country baseline
- **Aₚ** = Maximum pillar deviation
- **kₚ** = Saturation parameter

Normalization already bounds Δ.

The purpose of `tanh()` is **not** bounding.

Its purpose is introducing **diminishing returns**.

Large amounts of evidence gradually produce smaller additional changes instead of growing linearly forever.

---

# Step 5 — Positive / Negative Asymmetry

Different pillars may penalize negative evidence more strongly than they reward positive evidence.

Instead of

```
A_p
```

allow

\[
\boxed{
A_p^+
\qquad
A_p^-
}
\]

Example:

```
Positive ESG improvements

↓

Maximum swing +25

Negative ESG failures

↓

Maximum swing -35
```

This better reflects asymmetric ESG risk.

Whether separate values improve performance should be determined through calibration.

---

# Step 6 — Method Trust

The implementation already ranks evidence sources.

Example:

```
dataset_lookup

>

extracted

>

peer_ratio_fallback

>

coarse_bucket
```

Rather than introducing a new reliability variable, incorporate this directly into confidence.

Conceptually:

```
Adjusted Confidence

=

Confidence

×

Method Trust Multiplier
```

This reuses existing infrastructure and naturally gives greater influence to deterministic data sources than LLM-derived evidence.

---

# Parameter Calibration

None of the following should be hardcoded:

- A
- k
- β
- Evidence threshold
- Method trust multipliers
- Positive/negative asymmetry

All should be optimized using the existing calibration harness against benchmark datasets.

---

# Per-Pillar Calibration

Each ESG pillar should maintain independent parameters.

```
Environmental

A_E
k_E

Social

A_S
k_S

Governance

A_G
k_G
```

Different pillars have different evidence density and factor distributions.

Independent calibration preserves this behavior.

---

# Computational Complexity

Current implementation:

```
O(number of claims)
```

Proposed implementation:

```
O(number of claims)
```

Additional computations consist only of:

- normalization
- coverage calculation
- one tanh evaluation
- one evidence threshold check

The computational complexity remains unchanged and scales comfortably to 100,000+ companies.

---

# Final Formula

For each ESG pillar:

### Aggregate Evidence

\[
\boxed{
\Delta
=
\frac
{\sum_{i\in C}
w_i c_i \delta_i}
{\sum_{i\in C}
w_i}
}
\]

### Coverage

\[
\boxed{
Coverage
=
\frac
{\sum_{i\in C}
w_i}
{\sum_{i\in Pillar}
w_i}
}
\]

### Coverage Multiplier

\[
\boxed{
CoverageMultiplier
=
\beta
+
(1-\beta)
Coverage
}
\]

### Final Score

\[
\boxed{
Score
=
B
+
A_p
\tanh(k_p\Delta)
\times
CoverageMultiplier
}
\]

subject to

```
If EvidenceMass < Threshold

↓

Score = Baseline
```

---

# Summary

This proposal introduces the smallest practical mathematical improvement while remaining fully compatible with the current architecture.

It:

- Preserves deterministic scoring.
- Keeps the existing confidence and freshness mechanisms.
- Prevents excessive cross-factor accumulation.
- Rewards broader evidence coverage.
- Avoids overreacting to sparse or weak evidence.
- Reuses existing method trust instead of introducing duplicate reliability concepts.
- Supports asymmetric treatment of positive and negative evidence.
- Treats every new parameter as a calibration problem rather than a hardcoded assumption.
- Maintains explainability, auditability, and linear computational complexity.

The resulting Formula Estimator remains transparent and deterministic while providing a more realistic mathematical model of how ESG evidence accumulates across multiple independent factors.

---
## CHANGES 5.0
# Final ESG Formula Estimator (Production Recommendation)

## Design Goals

This formulation is designed to:

- Preserve the existing deterministic Formula Estimator architecture.
- Remain fully explainable and auditable.
- Scale efficiently to 100,000+ companies.
- Reuse existing pipeline components instead of introducing duplicate logic.
- Improve sparse-evidence handling without sacrificing differentiation.
- Be calibrated empirically rather than relying on manually chosen constants.

---

# Existing Pipeline

```
Signals
    ↓
Evidence Extraction
    ↓
Claims
    ↓
Formula Estimator
    ↓
Holistic Estimator
    ↓
Reconciliation
    ↓
Verification
```

No architectural changes are required.

---

# Existing Components Reused

The proposal intentionally builds on existing functionality.

### Already implemented

- Best claim selection per factor
- Evidence freshness decay (`evidence_freshness.py`)
- Peer Anchor
- `_METHOD_RANK`
- Country baselines
- Factor weights
- Calibration Harness

Nothing is duplicated.

---

# Step 0 — Method-Trust Adjusted Confidence

Reuse the existing `_METHOD_RANK` ordering.

Instead of introducing a new reliability variable, fold method trust into confidence.

\[
\boxed{
c_i' = c_i \times m(method_i)
}
\]

Example initial multipliers:

| Method | Multiplier |
|---------|-----------:|
| dataset_lookup | 1.00 |
| extracted | 0.90 |
| peer_ratio_fallback | 0.70 |
| coarse_bucket | 0.50 |

These values are **initial calibration values**, not fixed constants.

Freshness remains inside `cᵢ` exactly as implemented today.

No second decay term is introduced.

---

# Step 1 — Partition Contributions

Separate real company evidence from peer-derived prior information.

```
C_claims

↓

All registry factors
that produced claims

(best-claim-per-factor)

PA

↓

Peer Anchor
(if available)
```

The Peer Anchor is **not** a registry factor.

It represents prior information from comparable companies.

---

# Step 2 — Evidence Gate

Compute the confidence-weighted evidence mass.

\[
\boxed{
M
=
\sum_{i\in C_{claims}}
w_i c_i'
}
\]

If

```
M ≥ Threshold
```

then

```
Use:

Claims + Peer Anchor
```

Otherwise

```
Ignore weak claims

Use:

Peer Anchor only
```

This extends the existing invariant:

```
No evidence

↓

Baseline
```

to

```
No meaningful evidence

↓

Peer-informed baseline
```

This prevents a single weak claim from moving the score while preserving differentiation introduced by the Peer Anchor.

---

# Step 3 — Normalized Evidence Swing

Compute the weighted evidence direction.

\[
\boxed{
\Delta
=
\frac
{\sum_{i\in C}
w_i c_i' \delta_i}
{\sum_{i\in C}
w_i}
}
\]

where

```
C

=

Gate-passed claims

+

Peer Anchor (if present)
```

If no evidence remains,

```
Δ = 0
```

Using only factor weights in the denominator ensures confidence suppresses the contribution without artificially amplifying sparse low-confidence evidence.

---

# Step 4 — Coverage

Coverage should measure **how much real company evidence exists**, not how much peer information exists.

Therefore:

- Include registry claims.
- Exclude Peer Anchor.
- Exclude gated-out claims.

\[
\boxed{
Coverage
=
\frac
{\sum_{i\in C_{claims}}
w_i}
{\sum_{i\in Registry}
w_i}
}
\]

Properties:

\[
0
\le
Coverage
\le
1
\]

Coverage can never exceed 1 because the Peer Anchor is excluded.

---

# Step 5 — Coverage Multiplier

Sparse evidence should still move the score.

Introduce a coverage floor.

\[
\boxed{
CoverageMultiplier
=
\beta
+
(1-\beta)
\times
Coverage
}
\]

where

```
0 < β < 1
```

Example:

```
β = 0.6

Coverage = 0.25

↓

Multiplier

=

0.70
```

This rewards broader evidence without eliminating legitimate sparse evidence.

---

# Step 6 — Nonlinear Saturation

Compute the final pillar score.

Select

```
A+

if Δ ≥ 0

A-

if Δ < 0
```

Then

\[
\boxed{
Score
=
Clamp
\left(
B
+
A
\tanh(k\Delta)
\times
CoverageMultiplier,
0,
100
\right)
}
\]

Where

- **B** = Country baseline
- **A⁺ / A⁻** = Maximum positive / negative pillar deviation
- **k** = Saturation parameter

Normalization prevents unbounded growth.

`tanh()` introduces diminishing returns rather than additional bounding.

---

# Initial Parameters

Use values that reproduce the current design intent.

The factor registry already targets roughly

```
±30 point

maximum
```

pillar movement.

Starting values:

| Parameter | Initial Value |
|-----------|--------------:|
| A⁺ | 40 |
| A⁻ | 40 |
| k | 1.0 |
| β | 0.6 |
| Evidence Threshold | 2.5 |

Since

```
tanh(1)

≈

0.76
```

the effective maximum swing becomes

```
40 × 0.76

≈

30
```

matching the existing scoring philosophy.

---

# Calibration Strategy

Avoid tuning every parameter simultaneously.

## Phase 1

Freeze:

- β
- Evidence Threshold
- Method multipliers

Tune only:

- A
- k

for each pillar.

---

## Phase 2

Evaluate on a held-out validation set.

Accept only if:

- Environmental performance does not regress.
- Social performance does not regress.
- Governance performance does not regress.

using Spearman/Pearson correlation against benchmark ESG datasets.

---

## Phase 3

Once sufficient labeled data exists, optimize:

- β
- Method multipliers
- Positive/negative asymmetry
- Evidence threshold

This staged approach minimizes overfitting.

---

# Verification Tests

Before benchmark calibration, verify using deterministic synthetic cases.

### Test 1

```
No claims

No Peer Anchor

↓

Exactly Country Baseline
```

---

### Test 2

```
No claims

Peer Anchor only

↓

Small differentiated movement
```

---

### Test 3

```
One weak claim

↓

Evidence Gate fires

↓

Peer Anchor only
```

---

### Test 4

```
All factors strongly positive

↓

Approximately

Baseline +30
```

---

### Test 5

```
All factors strongly negative

↓

Approximately

Baseline -30
```

---

### Test 6

```
Balanced positive and negative evidence

↓

Δ ≈ 0

↓

Approximately Baseline
```

---

# Explicitly Rejected

The following ideas were evaluated but intentionally rejected:

- Confidence squaring (`c²`)
- Per-claim `tanh()`
- Separate source reliability variable (`rᵢ`)
- Additional exponential time decay (`e^{-λΔt}`)
- Continuous integral formulation

Each either duplicated existing functionality or solved a problem already addressed elsewhere in the pipeline.

---

# Final Formula

### Method-Adjusted Confidence

\[
c_i' = c_i \times m(method_i)
\]

---

### Evidence Mass

\[
M
=
\sum_{i\in C_{claims}}
w_i c_i'
\]

---

### Evidence Gate

```
If

M < Threshold

↓

Ignore claims

↓

Use Peer Anchor only
```

---

### Normalized Swing

\[
\boxed{
\Delta
=
\frac
{\sum_{i\in C}
w_i c_i' \delta_i}
{\sum_{i\in C}
w_i}
}
\]

---

### Coverage

\[
Coverage
=
\frac
{\sum_{i\in C_{claims}}
w_i}
{\sum_{i\in Registry}
w_i}
\]

---

### Coverage Multiplier

\[
CoverageMultiplier
=
\beta
+
(1-\beta)
Coverage
\]

---

### Final Score

\[
\boxed{
Score
=
Clamp
\left(
B
+
A
\tanh(k\Delta)
\times
CoverageMultiplier,
0,
100
\right)
}
\]

---

# Summary

This formulation represents a production-ready evolution of the existing Formula Estimator.

It:

- Preserves deterministic and explainable scoring.
- Reuses the existing freshness, peer-anchor, and method-ranking systems.
- Separates real company evidence from peer-derived priors.
- Prevents weak evidence from introducing noisy score changes.
- Rewards broader evidence coverage without penalizing sparse but valid information.
- Introduces smooth nonlinear saturation to model diminishing returns.
- Supports asymmetric positive and negative score movements where justified.
- Maintains linear computational complexity.
- Uses staged calibration to minimize overfitting.

The resulting estimator is mathematically grounded, implementation-friendly, and integrates cleanly with the current ESG scoring pipeline while preserving transparency and auditability.

---
## CHANGES 6.0
# Mathematical Audit of the ESG Pipeline

## Overall Philosophy

Each phase should solve exactly one mathematical problem.

Avoid performing the same operation twice (e.g., confidence scaling, reliability adjustment, or evidence decay).

The pipeline should progress like this:

Raw Evidence
    ↓
Evidence Quality
    ↓
Evidence Strength
    ↓
Company Score
    ↓
Global Reasoning
    ↓
Final Calibration

Each phase should contribute one transformation.

---

# Phase 1 — Signal Collection

## Purpose

Collect raw ESG-related signals from multiple structured and unstructured sources.

## Current Mathematics

Mostly deterministic filtering and ranking.

## Evaluation

This phase should remain almost entirely statistical rather than predictive.

Avoid introducing ESG scoring here.

The objective is only to answer:

- Is this information relevant?
- How trustworthy is the source?
- How recent is it?

## Recommendation

Keep mathematical complexity minimal.

Only compute metadata such as:

- Source credibility
- Publication date
- Document relevance
- Duplicate similarity

Do not compute ESG impact here.

Priority:
LOW

---

# Phase 2 — Evidence Extraction

## Purpose

Convert raw documents into structured ESG claims.

## Current Mathematics

Confidence scores
Freshness decay
Best-claim selection

## Evaluation

This phase already performs the correct mathematical operations.

Confidence should represent extraction certainty only.

Freshness should represent temporal reliability only.

These should not attempt to estimate ESG impact.

The current exponential freshness decay is appropriate.

Best-claim selection correctly prevents repeated news articles from artificially increasing influence.

## Suggested Improvements

Calibrate confidence.

LLM confidence values are rarely well calibrated.

Apply confidence calibration using held-out validation data.

Do not square confidence.

Do not apply nonlinear transforms.

Those belong later.

Priority:
MEDIUM

---

# Phase 3 — Formula Estimator

## Purpose

Convert extracted evidence into deterministic ESG scores.

## Evaluation

This is the mathematical core of the pipeline.

This is the correct place for:

- weighted aggregation
- normalization
- coverage estimation
- evidence gating
- nonlinear saturation

These operations should not appear elsewhere.

## Suggested Formula

Use the finalized v5 formulation.

It correctly separates:

- evidence quality
- evidence quantity
- evidence coverage
- score saturation

without duplicating concepts.

## Keep

- country baseline
- factor weights
- peer anchor
- confidence
- coverage
- tanh saturation
- evidence gate

## Do Not Add

- second freshness decay
- additional reliability term
- per-claim nonlinearities
- Bayesian updates
- confidence squaring

All of those duplicate work already performed upstream.

Priority:
VERY HIGH

---

# Phase 4 — Holistic Estimator

## Purpose

Estimate ESG characteristics not directly observable through deterministic evidence.

## Current Role

LLM reasoning.

## Evaluation

This phase should NOT repeat deterministic calculations.

Instead, it should answer questions like:

"What does the overall evidence imply?"

rather than

"How many carbon incidents occurred?"

Those questions were already answered.

## Suggested Improvement

Instead of predicting absolute ESG scores,

predict

Residual Score

Meaning

Estimate only what the Formula Estimator cannot explain.

Conceptually

Holistic

=

Residual

rather than

Holistic

=

Independent ESG Score

This reduces disagreement between the deterministic and holistic models.

Priority:
HIGH

---

# Phase 5 — Reconciliation

## Purpose

Combine deterministic and holistic estimates.

## Evaluation

Current weighted averaging is reasonable.

However,

weights should represent uncertainty rather than fixed importance.

The more deterministic evidence exists,

the less influence the holistic estimate should have.

Conversely,

if deterministic evidence is sparse,

the holistic estimate becomes more valuable.

## Suggested Improvement

Replace fixed blending weights with uncertainty-aware weighting.

Example:

High Formula Confidence

↓

Formula dominates

Low Formula Confidence

↓

Holistic dominates

This naturally adapts to evidence availability.

Priority:
HIGH

---

# Phase 6 — Verification

## Purpose

Detect unreasonable outputs.

## Evaluation

Currently focused on sanity checking.

This phase can become much stronger.

## Suggested Improvements

Compute:

- disagreement between Formula and Holistic
- uncertainty interval
- confidence score
- anomaly detection

Large disagreement should trigger manual review rather than silently averaging.

Example:

Formula

72

Holistic

28

↓

Review required

instead of

↓

Final

50

Priority:
HIGH

---

# Cross-Phase Observations

## Confidence

Confidence should only describe extraction certainty.

Do not reinterpret confidence later as reliability or importance.

---

## Freshness

Freshness should only modify evidence confidence once.

Never decay evidence twice.

---

## Coverage

Coverage belongs only inside the Formula Estimator.

It should never influence extraction or reconciliation.

---

## Peer Anchor

Peer Anchor is prior information.

It should never influence:

- evidence confidence
- coverage
- extraction

It should only influence deterministic scoring.

---

## Saturation

Nonlinear saturation (`tanh`) belongs only inside the Formula Estimator.

Applying nonlinearities elsewhere makes interpretation difficult.

---

## Explainability

Every mathematical operation should be attributable.

Every final score should be decomposable into:

Country Baseline
+
Peer Anchor
+
Evidence Contributions
+
Coverage Adjustment
+
Saturation
+
Reconciliation

No hidden transformations should exist outside these stages.

---

# Potential Future Improvements

Once a large benchmark dataset (10k+ labeled companies) becomes available, consider:

1. Learning factor weights rather than manually assigning them.
2. Learning saturation parameters per pillar.
3. Calibrating method-trust multipliers.
4. Bayesian uncertainty estimation for final scores.
5. Confidence intervals around pillar scores.
6. Learning reconciliation weights from benchmark data instead of manually tuning them.

These should be considered Phase 2 research improvements rather than immediate production changes.

---

# Overall Verdict

| Phase | Current State | Recommendation |
|--------|---------------|---------------|
| Signal Collection | Good | Keep simple and deterministic |
| Evidence Extraction | Good | Calibrate confidence only |
| Formula Estimator | Most critical | Implement finalized v5 formulation |
| Holistic Estimator | Good concept | Predict residuals instead of full scores |
| Reconciliation | Good | Make blending uncertainty-aware |
| Verification | Needs expansion | Add disagreement detection and uncertainty estimation |

The pipeline architecture itself is strong. Most future improvements should focus on better calibration, uncertainty modeling, and phase separation rather than introducing more complex mathematics. Each phase should remain responsible for a single mathematical transformation, keeping the system modular, interpretable, and scalable.

One additional insight that emerged after reviewing the entire design is that your pipeline naturally resembles a Bayesian estimation process, even though it isn't explicitly implemented as one:

- Signal Collection builds the evidence.
- Evidence Extraction estimates the likelihood of observations.
- Formula Estimator computes a deterministic posterior from structured evidence.
- Holistic Estimator provides a learned prior/residual estimate.
- Reconciliation combines these estimates into a final posterior.
- Verification performs posterior consistency checks.

Thinking about the pipeline in these terms can help guide future improvements: rather than adding more formulas, focus on making each stage represent a distinct step in the estimation process with well-defined inputs, outputs, and uncertainty.

---
## CHANGES 7.0
# Recommended Improvement — Reconciliation Confidence

## Current Implementation

Formula confidence is currently estimated using the number of contributing factors.

Conceptually:

```
FormulaConfidence
=
f(number_of_contributions)
```

This correctly captures that more evidence generally implies higher confidence.

However, it assumes that every contribution carries equal evidential value.

For example:

- one contribution with confidence = 0.90
- one contribution with confidence = 0.15

both increase confidence equally.

This no longer matches the mathematical formulation introduced in Formula v5.

---

## Proposed Improvement

Reuse the evidence mass already computed inside the Formula Estimator.

\[
M
=
\sum_i
w_i
c_i'
\]

where

- \(w_i\) = factor weight
- \(c_i'\) = method-adjusted confidence

Instead of deriving Formula confidence from contribution count, derive it from evidence mass.

Conceptually:

```
FormulaConfidence
=
f(EvidenceMass)
```

rather than

```
FormulaConfidence
=
f(NumberOfContributions)
```

---

## Why This Is Better

Evidence mass captures:

- number of claims,
- claim confidence,
- factor importance,
- method reliability.

Contribution count captures only:

- how many claims survived.

Two companies with five claims should not receive identical Formula confidence if one has five high-confidence dataset-derived claims and the other has five weak LLM-extracted claims.

Evidence mass naturally distinguishes between them.

---

## Advantages

- Reuses an existing quantity already computed by Formula v5.
- Introduces no additional mathematical concepts.
- Better represents deterministic evidence quality.
- Produces a more principled confidence estimate for reconciliation.
- Remains fully deterministic and explainable.
- Requires only a localized change to the reconciliation stage.

---

## Expected Effect

Companies with:

- many strong claims

will naturally receive higher Formula confidence.

Companies with:

- many weak claims

will no longer appear equally trustworthy simply because they have the same number of contributions.

This should improve reconciliation by allowing the Formula estimator to dominate only when it is supported by genuinely strong evidence, rather than merely numerous evidence.

---

## Priority

**High**

This is a low-complexity change that aligns reconciliation with the mathematical assumptions already introduced in Formula v5.

Unlike broader architectural changes, it requires no redesign and is directly testable through the existing calibration harness.

---
## CHANGES 8.0
# Final Recommendation — Formula Confidence in Reconciliation

## Problem

Formula v5 introduces the evidence mass

\[
M
=
\sum_i
w_i c_i'
\]

This quantity is used by the Formula Estimator to decide whether claim evidence is sufficient.

However, Reconciliation asks a different question:

> "How trustworthy is the Formula estimator?"

These two questions should not necessarily use the same definition of evidence mass.

---

# Evidence Mass Should Have Two Definitions

## 1. Claim Evidence Mass (Formula Estimator)

Used only inside the Formula Estimator.

\[
\boxed{
M_{claims}
=
\sum_{i\in C_{claims}}
w_i c_i'
}
\]

Purpose:

- Evidence Gate
- Claim validation
- Formula scoring

The Peer Anchor is intentionally excluded because it is prior information rather than direct company evidence.

This definition should remain unchanged.

---

## 2. Formula Trust Mass (Reconciliation)

Used only inside Reconciliation.

\[
\boxed{
M_{trust}
=
M_{claims}
+
w_{PA} c_{PA}
}
\]

where

- \(w_{PA}\) is the Peer Anchor weight
- \(c_{PA}\) is the Peer Anchor confidence

Purpose:

Estimate how trustworthy the entire deterministic Formula estimate is.

Unlike the Evidence Gate, this should include the Peer Anchor because peer-derived statistics genuinely increase confidence in the Formula estimate compared to using only the country baseline.

---

# Formula Confidence Function

Replace contribution-count confidence with a continuous function of trust mass.

\[
\boxed{
c_f
=
\min
\left(
1,\;
0.4
+
0.04\,M_{trust}
\right)
}
\]

Properties:

- Minimum confidence remains 0.4.
- Confidence increases smoothly with stronger deterministic evidence.
- Full confidence is reached around \(M_{trust}=15\).
- High-quality claims naturally produce higher confidence than numerous weak claims.

---

# Why This Is Better

The previous implementation assumed

```
5 weak claims

≈

5 strong claims
```

because only the number of contributions mattered.

The new formulation distinguishes

```
Contribution Count

↓

Quantity
```

from

```
Evidence Mass

↓

Quantity × Quality
```

making Formula confidence consistent with the mathematical assumptions introduced by Formula v5.

---

# Continuity with Existing Behaviour

One important edge case is preserved.

### Peer Anchor Only

Current implementation

```
Formula Confidence

≈

0.55
```

Proposed implementation

```
Mtrust

≈

10 × 0.5

=

5

↓

cf

=

0.4 + 0.04 × 5

=

0.60
```

The behaviour remains close to the existing system.

This avoids unintentionally weakening the Formula estimator for sparse-evidence companies, where the Peer Anchor is specifically intended to provide meaningful prior information.

---

# Design Principle

The key insight is that evidence mass serves two different mathematical purposes.

Inside Formula Estimation:

```
Evidence Mass

↓

Can the claims themselves be trusted?
```

Inside Reconciliation:

```
Evidence Mass

↓

Can the deterministic estimator as a whole be trusted?
```

These are different questions and therefore justify different definitions of evidence mass.

Keeping these definitions separate preserves mathematical consistency while avoiding regressions in sparse-evidence scenarios.

---

# Recommendation

Adopt two explicit quantities throughout the codebase:

- **M_claims** — used exclusively by the Formula Estimator and Evidence Gate.
- **M_trust** — used exclusively by the Reconciliation stage to derive Formula confidence.

This removes ambiguity, preserves current behaviour where desirable, and aligns every phase with its intended mathematical role.