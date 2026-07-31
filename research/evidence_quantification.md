# Research Notes: The Signal Quantification Problem in ESG Estimation

## Background

Our ESG estimation pipeline is capable of collecting and structuring ESG evidence from multiple heterogeneous sources, including:

- News articles
- Sustainability reports
- SEC filings
- Government databases
- Certifications
- NGO reports
- Company disclosures

The pipeline can reliably determine:

- ESG Pillar (E / S / G)
- ESG Factor
- Polarity (Positive / Neutral / Negative)

However, it cannot objectively determine **how much** each piece of evidence should influence the ESG score.

---

# The Fundamental Problem

Consider an example.

```
Evidence

↓

Oil spill
```

The pipeline correctly identifies:

```
Environmental

↓

Negative
```

But then comes the difficult question:

```
How negative?
```

Should it contribute

```
-0.15 ?
-0.42 ?
-0.91 ?
```

There is currently no objective way to answer this.

---

# Why This Problem Exists

Our benchmark ESG scores are obtained from providers such as:

- BCorp
- Upright Platform

These providers disclose only the **final ESG score**.

They do **not** disclose:

- evidence considered
- evidence weights
- intermediate scores
- feature engineering
- scoring equations
- contribution of individual evidence

Our data therefore looks like:

```
Evidence
      ↓
Unknown Proprietary Model
      ↓
Final ESG Score
```

The middle step is completely hidden.

Therefore we have **no evidence-level ground truth**.

---

# Initial Thought

The original idea was to convert qualitative evidence into a numerical value using statistical normalization.

Example:

```
Oil Spill

↓

Extract Features

↓

Normalize

↓

Contribution
```

The assumption was that normalization techniques such as percentiles could be used.

Example:

```
3200 barrels

↓

94th percentile
```

---

# Why This Does NOT Work

This assumption breaks under the actual pipeline architecture.

Our pipeline processes companies independently.

Example:

```
Company A

↓

Oil Spill
```

```
Company B

↓

Board Diversity
```

```
Company C

↓

Labor Strike
```

These companies rarely contain the same evidence types.

There is no dataset such as

```
10,000 Oil Spills

↓

Distribution

↓

Percentile
```

for every ESG factor.

Without a common reference distribution, percentile normalization is impossible.

---

# A Second Realization

Initially the research question was framed as

> "How do we convert qualitative evidence into numbers?"

This is actually the wrong question.

The real challenge is

> **How do we determine the impact of evidence on an ESG score when no evidence-level supervision exists?**

These are fundamentally different problems.

---

# Why Most Machine Learning Methods Fail

Many common ML techniques appear attractive but cannot solve this problem because they require supervision.

Examples include:

- Linear Regression
- Neural Networks
- Learning-to-Rank
- Bayesian Regression
- Gradient Boosting
- Latent Variable Models
- Structural Equation Modeling
- Item Response Theory

All of these require some form of

```
Evidence

↓

Known Target
```

For example

```
Oil Spill

↓

Known Severity = -0.72
```

or

```
Labor Strike

↓

Known Contribution = -0.18
```

No such labels exist.

Therefore these methods cannot be trained correctly.

This is an **information limitation**, not an algorithm limitation.

---

# Evaluating Possible Approaches

## 1. Rule-Based Mapping

Example

```
Oil Spill

↓

Severity = -0.8
```

### Advantages

- Simple
- Explainable

### Problems

- Completely hand-designed
- Requires extensive tuning
- Difficult to justify scientifically

**Verdict:** Not recommended.

---

## 2. LLM Severity Estimation

Example

```
LLM

↓

Severity = 0.87
```

### Advantages

- Easy to implement
- Flexible

### Problems

- Subjective
- Non-deterministic
- Poor reproducibility
- Difficult to validate

**Verdict:** Not suitable for a deterministic ESG scoring pipeline.

---

## 3. Clustering

Example

```
Evidence

↓

Embedding

↓

Clusters
```

### What clustering actually does

Clustering groups similar evidence together.

Example

```
Cluster 1

Large Oil Spills

------------

Cluster 2

Water Pollution

------------

Cluster 3

Illegal Waste Disposal
```

This is useful because similar evidence is grouped automatically.

### What clustering does NOT do

Clustering does **not** determine impact.

It cannot answer

```
Cluster 1

↓

-12 ESG points
```

It only discovers similarity.

Impact estimation remains unsolved.

**Verdict:** Useful as an evidence organization technique, **not** as a complete qualitative-to-quantitative conversion method.

---

## 4. Embeddings

Embeddings convert evidence into dense numerical vectors.

Example

```
Oil Spill

↓

Embedding Vector
```

This preserves semantic similarity.

However, embeddings still do not indicate

```
How much should this change the ESG score?
```

They solve representation, not measurement.

**Verdict:** Useful for retrieval, similarity search and clustering, but not for contribution estimation.

---

## 5. Measurement Models

Another possibility is to build dedicated measurement functions for each ESG factor.

Example

```
Oil Spill

↓

{
Volume,
Fine,
Protected Area,
Investigation,
Cleanup Cost
}

↓

Pollution Impact
```

Unlike universal conversion techniques, these models are factor-specific.

Advantages:

- Explainable
- Auditable
- Compatible with deterministic formulas

Limitations:

- Require domain expertise
- Still need careful calibration
- No objective ground truth for validation

**Verdict:** The most scientifically defensible approach among deterministic methods, but still requires expert-designed measurement functions.

---

# What Clustering Can Actually Contribute

Clustering should not be viewed as the solution.

Instead, it can support the pipeline by organizing similar evidence.

```
Evidence

↓

Embedding

↓

Clusters

↓

Evidence Organization
```

The cluster identity becomes another feature available to downstream modules.

It does **not** produce an ESG contribution.

---

# The Real Pipeline Gap

Our pipeline already solves:

✔ Evidence collection

✔ Evidence extraction

✔ ESG factor classification

✔ Polarity detection

✔ Confidence estimation

✔ Formula aggregation

The missing component is:

```
Evidence

↓

Impact Measurement

↓

Formula Contribution
```

This is the only unresolved stage.

---

# Current Understanding

After evaluating the available techniques, the following conclusions emerge.

### Clustering

✔ Good for discovering similar evidence.

✘ Does not estimate impact.

---

### Embeddings

✔ Good numerical representation.

✘ No notion of contribution magnitude.

---

### Rule-Based Systems

✔ Transparent.

✘ Arbitrary and heavily tuned.

---

### LLM Severity

✔ Flexible.

✘ Subjective and non-reproducible.

---

### Supervised Machine Learning

✔ Powerful.

✘ Impossible without evidence-level labels.

---

### Latent Variable Models

✔ Scientifically rigorous.

✘ Require observed supervision that does not exist.

---

# Final Conclusion

There does **not** appear to be a general-purpose qualitative-to-quantitative conversion algorithm capable of objectively assigning ESG contribution magnitudes under the current data constraints.

The limitation is not computational.

It is informational.

The available benchmark datasets expose only the **final ESG score**, while hiding the internal evidence contributions.

As a result:

- No supervised learning approach can recover evidence impacts.
- Clustering can organize evidence but cannot quantify it.
- Embeddings improve representation but not measurement.
- Rule-based systems require subjective assumptions.
- LLMs provide subjective estimates rather than objective measurements.

Therefore, the research problem should be reframed.

Instead of asking:

> **"How can qualitative evidence be converted into numbers?"**

the more accurate research question is:

> **"How can objective, transparent, and auditable evidence impact functions be designed when evidence-level supervision is unavailable?"**

This shifts the focus from searching for a universal conversion algorithm to designing scientifically defensible measurement models that integrate naturally with the deterministic Formula Estimator.

This appears to be the primary unresolved research challenge within the current ESG estimation pipeline.