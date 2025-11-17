# Experimental Design Types Guide

This guide explains the 6 design types available in the Protein Stability DoE Designer and when to use each one.

---

## Overview

The Designer supports 6 experimental design types:

| Design Type | Best For | Sample Size | Factors |
|-------------|----------|-------------|---------|
| **Full Factorial** | Complete exploration | All combinations | 2-4 factors |
| **Latin Hypercube (LHS)** | Large factor spaces | User-defined | 3+ factors |
| **Fractional Factorial** | Efficient screening | 2^(k-p) runs | 4+ factors (2 levels) |
| **Plackett-Burman** | Ultra-efficient screening | N+1 runs | 5+ factors |
| **Central Composite (CCD)** | Response surface optimization | 2^k + 2k + cp | 2-5 numeric factors |
| **Box-Behnken** | Optimization without corners | Moderate | 3+ numeric factors |

---

## 1. Full Factorial Design

### Description
Tests **all possible combinations** of factor levels. The gold standard for complete exploration.

### When to Use
- **2-4 factors** with 2-3 levels each
- You need to understand **all interactions**
- Sample capacity allows (≤96 wells per plate)
- Budget and time permit

### Formula
**Sample size = L₁ × L₂ × L₃ × ... × Lₖ**
- L = number of levels per factor
- k = number of factors

### Example
**Factors:**
- NaCl: [100, 200] mM (2 levels)
- Glycerol: [5, 10, 15]% (3 levels)
- pH: [7.0, 8.0] (2 levels)

**Result:** 2 × 3 × 2 = **12 combinations**

### Pros
✓ Complete coverage of experimental space
✓ Can detect all main effects and interactions
✓ Straightforward interpretation

### Cons
✗ Exponential growth (5 factors × 3 levels = 243 runs!)
✗ Inefficient for >4 factors
✗ May exceed plate capacity

### Use Case
> "I have 3 factors (pH, NaCl, glycerol) with 2-3 levels each and want to understand how they interact to affect protein stability."

---

## 2. Latin Hypercube Sampling (LHS)

### Description
**Space-filling design** that samples the experimental space efficiently. Each factor level appears exactly once in each stratified interval.

### When to Use
- **5+ factors** or many levels
- Limited budget (want specific sample size like 96)
- Exploring large parameter spaces
- Building surrogate models or ML

### Parameters
- **Sample Size**: Number of experiments (e.g., 96 for a full plate)
- **Optimization**: Optional SMT maximin for better space-filling

### Example
**Factors:**
- NaCl: [50, 100, 150, 200, 250] mM (5 levels)
- Glycerol: [0, 5, 10, 15, 20]% (5 levels)
- MgCl₂: [0, 1, 2, 5, 10] mM (5 levels)
- pH: [6.0, 7.0, 8.0, 9.0] (4 levels)

**Full factorial would be:** 5 × 5 × 5 × 4 = **500 runs** ❌

**LHS with 96 samples:** **96 runs** ✓

### Pros
✓ Fixed sample size (fits 96-well plate perfectly)
✓ Efficient for many factors
✓ Good space coverage
✓ Works with mixed numeric/categorical factors

### Cons
✗ May miss specific interactions
✗ Categorical factors use cycling (not true LHS)
✗ Requires advanced analysis (GAM, Gaussian Process)

### Optimization Options
- **Standard (pyDOE3)**: Uses 'center' criterion
- **Optimized (SMT)**: Uses 'maximin' criterion for better space-filling

### Use Case
> "I have 6 factors to screen and only 96 wells available. I want to explore the entire parameter space efficiently."

---

## 3. 2-Level Fractional Factorial

### Description
Uses a **fraction** of the full factorial design by confounding higher-order interactions. Assumes some interactions are negligible.

### When to Use
- **4-7 factors** with **2 levels each** (low/high)
- Screening phase to identify important factors
- Interactions expected to be small
- Limited resources

### Resolution Levels

| Resolution | Confounding | Best For |
|------------|-------------|----------|
| **III** | Main effects confounded with 2-way interactions | Initial screening, assume no interactions |
| **IV** | Main effects clear, 2-way interactions confounded | Moderate screening |
| **V** | Main effects + 2-way interactions clear | Detailed screening |

### Example
**Factors (2 levels each):**
- NaCl: [100, 500] mM
- Glycerol: [0, 20]%
- pH: [6.5, 8.5]
- MgCl₂: [0, 10] mM
- Detergent: [None, 0.1%]

**Full factorial:** 2⁵ = **32 runs**
**Fractional (Resolution IV):** **16 runs** (50% reduction)

### Pros
✓ 50-75% fewer runs than full factorial
✓ Main effects always estimable
✓ Works well for screening
✓ Statistically efficient

### Cons
✗ Only works with 2 levels per factor
✗ Some interactions may be confounded
✗ Need to convert multi-level factors to 2 levels

### Use Case
> "I have 5 factors and want to quickly identify which ones significantly affect stability before doing a detailed study."

---

## 4. Plackett-Burman Design

### Description
**Ultra-efficient screening** design for identifying important factors from many candidates. Uses orthogonal arrays.

### When to Use
- **5-15 factors** to screen
- Only need to identify **main effects**
- Assume interactions are negligible
- Very limited budget

### Formula
**N = 4k runs** (where k is integer, e.g., 12, 20, 24, 28...)

For **n factors**, uses **n+1 runs** (minimum)

### Example
**Screening 7 factors:**
- NaCl, KCl, MgCl₂, CaCl₂, Glycerol, DMSO, pH

**Full factorial (2⁷):** **128 runs** ❌
**Plackett-Burman:** **8 runs** ✓ (93% reduction!)

### Pros
✓ Extremely efficient (n+1 runs for n factors)
✓ Good for initial screening
✓ Orthogonal design (clean main effects)

### Cons
✗ Main effects ONLY (no interactions)
✗ Must assume interactions are negligible
✗ Only 2 levels per factor
✗ Complex aliasing structure

### Use Case
> "I have 10 potential buffer additives and want to quickly find the 2-3 most important ones before running a detailed study."

---

## 5. Central Composite Design (CCD)

### Description
**Response surface design** for optimization. Combines factorial points, axial (star) points, and center points.

### When to Use
- **2-5 numeric factors**
- Optimization phase (after screening)
- Want to fit quadratic models
- Looking for optimal conditions

### Design Structure
- **Factorial points**: 2^k corners (low/high)
- **Axial points**: 2k star points (on axes)
- **Center points**: Replicates for error estimation

### CCD Types

| Type | α (alpha) | Description | Use When |
|------|-----------|-------------|----------|
| **Faced** | α = 1 | Star points on cube faces | Standard, practical ranges |
| **Inscribed** | α < 1 | Design fits in cube | Factor limits are strict |
| **Circumscribed** | α > 1 | Star points outside cube | Can test beyond limits |

### Example
**Factors (3 levels each):**
- NaCl: [100, 250, 400] mM
- Glycerol: [0, 10, 20]%

**Full factorial (3²):** 9 runs
**CCD (faced):** **13 runs** (2² factorial + 2×2 axial + 5 center)

### Pros
✓ Fits quadratic models (finds optimum)
✓ More efficient than 3-level factorial
✓ Can identify curvature in response
✓ Center points estimate error

### Cons
✗ Only for numeric factors
✗ Requires 3+ levels worth of range
✗ More runs than screening designs
✗ Assumes smooth response surface

### Use Case
> "I've identified NaCl and glycerol as important. Now I want to find the optimal concentrations for maximum stability."

---

## 6. Box-Behnken Design

### Description
**Response surface design** that avoids extreme corners. Uses mid-points of edges instead.

### When to Use
- **3+ numeric factors**
- Optimization phase
- **Extreme conditions are unsafe/impractical**
- Want to avoid testing all high or all low combinations
- Fitting quadratic models

### Design Structure
- Tests points at **midpoints of edges** of the factor space
- Does **NOT test corner points**
- Includes center point replicates

### Example
**Factors:**
- NaCl: [100, 250, 400] mM (L, M, H)
- Glycerol: [0, 10, 20]% (L, M, H)
- pH: [6.5, 7.5, 8.5] (L, M, H)

**Combinations tested:**
- (L, L, M), (L, H, M), (H, L, M), (H, H, M) - pH at middle
- (L, M, L), (L, M, H), (H, M, L), (H, M, H) - Glycerol at middle
- (M, L, L), (M, L, H), (M, H, L), (M, H, H) - NaCl at middle
- (M, M, M) - Center point (replicated)

**Total: ~15 runs** (vs 27 for 3³ factorial)

### Pros
✓ No extreme corner combinations (safer)
✓ Fewer runs than CCD for 3+ factors
✓ Fits quadratic models
✓ Efficient for optimization

### Cons
✗ Requires ≥3 factors
✗ Only for numeric factors
✗ Can't test extreme combinations (if you need them)
✗ Slightly less efficient than CCD for 2 factors

### Use Case
> "I want to optimize 4 factors, but testing all factors at their extreme high values simultaneously might denature my protein."

---

## Decision Tree: Which Design to Use?

```
START
│
├─ Screening phase (identify important factors)?
│  ├─ YES → How many factors?
│  │        ├─ 4-7 factors → Fractional Factorial (Resolution IV)
│  │        ├─ 8-15 factors → Plackett-Burman
│  │        └─ 5+ factors, need flexibility → LHS (96 samples)
│  │
│  └─ NO → Optimization phase (find best conditions)?
│           ├─ Can test extreme combinations?
│           │   ├─ YES → Central Composite Design (CCD)
│           │   └─ NO → Box-Behnken Design
│           │
│           └─ 2-4 factors, want all interactions → Full Factorial
```

---

## Practical Recommendations

### For Protein Stability Studies

**Phase 1: Screening (Find Important Factors)**
- Start with **Plackett-Burman** or **Fractional Factorial**
- Test 5-10 potential buffer components
- Use 2 levels each (low/high)
- Goal: Identify 2-4 key factors

**Phase 2: Optimization (Find Best Conditions)**
- Use **CCD** or **Box-Behnken** with identified factors
- Use 3-5 levels per factor
- Fit quadratic model
- Find maximum stability conditions

**Phase 3: Validation**
- Run **Full Factorial** with final 2-3 factors
- Confirm interactions
- Build detailed understanding

### Sample Size Guidelines

| Factors | Full Factorial (3 levels) | LHS | Fractional/PB | CCD/BB |
|---------|---------------------------|-----|---------------|---------|
| 2 | 9 | 16-32 | N/A | 9-13 |
| 3 | 27 | 32-64 | 8-16 | 13-15 |
| 4 | 81 | 48-96 | 16-32 | 25-31 |
| 5 | 243 ❌ | 64-96 | 16-32 | N/A |
| 6 | 729 ❌ | 96 | 32-64 | N/A |

---

## Statistical Power Considerations

### Minimum Replicates
- **Screening designs**: No replicates needed (use center points in CCD/BB)
- **Optimization**: 3-5 center point replicates for error estimation
- **Full factorial**: 2-3 full replicates if budget allows

### Degrees of Freedom
Make sure you have enough runs for your model:
- **Linear model**: k + 1 parameters (k factors + intercept)
- **Interactions**: k + (k choose 2) + 1
- **Quadratic**: k + (k choose 2) + k + 1

**Rule of thumb:** Need at least **3× more runs than parameters** to estimate

---

## Advanced Tips

### Mixing Categorical and Numeric Factors
- **Full Factorial**: Works naturally
- **LHS**: Categoricals use cycling (acceptable)
- **Fractional/PB**: Treat categoricals as 2-level factors
- **CCD/BB**: Only for numeric factors (handle categoricals separately)

### Dealing with Constraints
Some factor combinations may be invalid:
- Example: "If detergent = None, then detergent_concentration must be 0"

The Designer automatically filters invalid combinations for:
- buffer pH + buffer_concentration
- detergent + detergent_concentration
- reducing_agent + reducing_agent_concentration

### Space-Filling vs. Structured Designs
- **Space-filling (LHS)**: Better for complex response surfaces, ML models
- **Structured (Factorial, CCD)**: Better for ANOVA, interaction analysis

---

## References

1. Box, G. E. P., Hunter, W. G., & Hunter, J. S. (2005). *Statistics for Experimenters*. Wiley.
2. Montgomery, D. C. (2017). *Design and Analysis of Experiments* (9th ed.). Wiley.
3. McKay, M. D., Beckman, R. J., & Conover, W. J. (1979). "A Comparison of Three Methods for Selecting Values of Input Variables". *Technometrics*, 21(2).
4. Plackett, R. L., & Burman, J. P. (1946). "The Design of Optimum Multifactorial Experiments". *Biometrika*, 33(4).

---

## Need Help?

- **Can't decide?** Start with LHS (96 samples) - it works for most cases
- **Unsure about factor levels?** Use 2-3 levels initially, can always add more
- **Too many combinations?** Switch to LHS or Fractional Factorial
- **Need optimization?** Use CCD after screening

For more details on implementation, see [README.md](../README.md) and [WELL_MAPPING.md](WELL_MAPPING.md).
