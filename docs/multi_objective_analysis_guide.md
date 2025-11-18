# Multi-Objective Analysis Guide

## Overview

The protein stability toolkit now supports **multi-response Design of Experiments (DoE) analysis** and **multi-objective Bayesian Optimization (BO)**. This allows you to analyze and optimize multiple conflicting objectives simultaneously.

## Table of Contents

1. [How Multi-Response Analysis Works](#how-multi-response-analysis-works)
2. [How Multi-Objective Bayesian Optimization Works](#how-multi-objective-bayesian-optimization-works)
3. [What You Can Control](#what-you-can-control)
4. [Understanding Pareto Frontiers](#understanding-pareto-frontiers)
5. [Limitations and Constraints](#limitations-and-constraints)
6. [Examples](#examples)

---

## How Multi-Response Analysis Works

### What is Multi-Response DoE?

In traditional DoE, you analyze **one response variable** at a time (e.g., just Tm). Multi-response DoE analyzes **multiple response variables simultaneously** from the same experimental data (e.g., Tm, Aggregation, and Activity).

### How Does It Work?

1. **Independent Models**: Each response gets its own **separate regression model**
   - Tm might use a quadratic model
   - Aggregation might use a linear model
   - Activity might use an interaction model

2. **Model Comparison**: For each response, the software:
   - Fits all 5 model types (mean, linear, interactions, quadratic, reduced)
   - Compares them using R², AIC, BIC
   - Selects the best model OR uses your manually selected model

3. **Statistical Analysis**: Each response gets:
   - Full ANOVA table
   - R-squared and model statistics
   - Main effects calculations
   - Factor significance testing

### Example Workflow

```
Data: Tm, Aggregation, Activity (3 responses)
      ↓
Compare Models for Each:
   Tm         → Best: Quadratic (R² = 0.89)
   Aggregation → Best: Linear (R² = 0.76)
   Activity   → Best: Interactions (R² = 0.82)
      ↓
Analyze Each Separately:
   Tm: pH optimal at 7.5, Glycerol increases Tm
   Aggregation: NaCl increases aggregation
   Activity: pH optimal at 8.0, Zinc required
```

**Key Point**: The responses are analyzed **independently**. They don't affect each other's regression models.

---

## How Multi-Objective Bayesian Optimization Works

### What is Multi-Objective Optimization?

When you have **conflicting objectives** (e.g., maximize Tm while minimizing Aggregation), there's no single "best" solution. Instead, there's a set of **trade-off solutions** called the **Pareto frontier**.

### The Pareto Frontier

**Definition**: A set of solutions where:
- You **cannot improve one objective without making another worse**
- These are the "non-dominated" solutions

**Example**:
```
Goal: Maximize Tm, Minimize Aggregation

Point A: Tm = 55°C, Aggregation = 8%
Point B: Tm = 52°C, Aggregation = 4%
Point C: Tm = 48°C, Aggregation = 12%

Pareto Frontier: A and B
  - A has higher Tm but more aggregation
  - B has lower aggregation but lower Tm
  - Both are valid trade-offs

Not Pareto: C (dominated by both A and B)
```

### How It Works Internally

1. **Multi-Objective BO Model**:
   - Uses Ax Platform's multi-objective capabilities
   - Builds a **Gaussian Process (GP) model** for each objective
   - Models learn correlations between factor settings and responses

2. **Pareto Frontier Extraction**:
   - Ax identifies all non-dominated points from your data
   - Returns parameters and objective values for each point
   - These represent the best trade-offs you've found so far

3. **Visualization**:
   - **2 objectives**: 2D scatter plot (X = Objective 1, Y = Objective 2)
   - **3 objectives**: 3D scatter plot
   - **>3 objectives**: No visualization (too high-dimensional)

### Multi-Objective vs Single-Objective

| Aspect | Single-Objective | Multi-Objective |
|--------|-----------------|-----------------|
| **Goal** | Find THE best point | Find trade-off solutions |
| **Output** | One optimal value | Pareto frontier (set of solutions) |
| **Plots** | Acquisition function, uncertainty | Pareto frontier scatter |
| **Suggestions** | Highest expected improvement | Balanced exploration of trade-offs |

---

## What You Can Control

### 1. Response Selection

**You Choose**:
- Which columns are responses (check boxes in GUI)
- How many responses to analyze (1 to any number)

**Example**:
```
Available columns: Tm, Aggregation, Activity, Kd
Selected responses: ✓ Tm, ✓ Aggregation, ✗ Activity, ✗ Kd
→ Analyzes Tm and Aggregation only
```

### 2. Optimization Direction (Per Response)

**You Choose** for each response:
- **Maximize**: Higher is better (e.g., Tm, Activity)
- **Minimize**: Lower is better (e.g., Aggregation, Cost)

**Example**:
```
Tm: Maximize ↑ (want higher melting temperature)
Aggregation: Minimize ↓ (want less aggregation)
Activity: Maximize ↑ (want higher enzymatic activity)
```

**Important**: The direction only affects **Bayesian Optimization suggestions**, not the statistical analysis. The DoE analysis treats all responses neutrally.

### 3. Model Type (Per Response)

**You Choose** for each response:
- **Auto**: Software selects best model based on R², AIC, BIC
- **Manual**: You force a specific model type (linear, quadratic, etc.)

**Example in Multi-Response**:
```
Tm: Auto → Selected Quadratic (R² = 0.89)
Aggregation: Auto → Selected Linear (R² = 0.76)
Activity: Manual → Force Interactions model
```

**Note**: You set ONE model type choice for all responses (e.g., "Auto" or "Linear"), but each response gets its own fitted model.

### 4. What You CANNOT Control

**You cannot**:
- ❌ "Combine" multiple responses into one metric
- ❌ Assign weights to responses (e.g., "Tm is 2x more important than Aggregation")
- ❌ Set target values (e.g., "Tm must be > 50°C")
- ❌ Constrain the Pareto frontier (all constraints must be in your data)

**Why?**: This is a limitation of the current implementation. The multi-objective BO treats all objectives equally and finds all Pareto-optimal solutions.

---

## Understanding Pareto Frontiers

### 2D Pareto Frontier (2 Responses)

**What You See**:
```
     Tm (°C)
        ↑
     60 |           * *    ← Pareto Frontier
        |        *   *        (Best trade-offs)
     55 |     *       *
        |  *     •  •  •   ← Dominated points
     50 |   • •   •          (Sub-optimal)
        +─────────────────→
         0    5   10   15   Aggregation (%)
```

**How to Read It**:
- **Stars (*)**: Pareto frontier points - best trade-offs
- **Dots (•)**: Dominated points - sub-optimal
- **Left-most point**: Lowest aggregation (but lower Tm)
- **Top point**: Highest Tm (but higher aggregation)
- **Middle points**: Balanced trade-offs

### 3D Pareto Frontier (3 Responses)

**What You See**:
- 3D scatter plot with Pareto points marked as stars
- Can rotate to view from different angles
- More complex to interpret but shows 3-way trade-offs

**Example**:
```
Objectives: Tm (↑), Aggregation (↓), Activity (↑)

Pareto Point A: High Tm, Low Aggregation, Medium Activity
Pareto Point B: Medium Tm, Medium Aggregation, High Activity
Pareto Point C: Medium Tm, Low Aggregation, Low Activity

All three are valid - depends on your priorities!
```

### More Than 3 Responses

**Limitation**: Cannot visualize >3D Pareto frontier

**What Happens**:
- Multi-objective BO **still works** internally
- Pareto frontier is **still calculated**
- But **no plot is shown** (can't visualize 4D+)
- You can see Pareto points in the **Recommendations tab** as text

**Recommendations Tab Shows**:
```
📊 PARETO-OPTIMAL SOLUTIONS (4 objectives)

Solution #1:
  Parameters: pH=7.5, NaCl=50mM, Glycerol=10%
  Objectives:
    Tm = 55.2°C (maximize)
    Aggregation = 6.8% (minimize)
    Activity = 82.1% (maximize)
    Stability = 7.2 days (maximize)

Solution #2:
  Parameters: pH=8.0, NaCl=25mM, Glycerol=15%
  Objectives:
    Tm = 52.8°C (maximize)
    Aggregation = 4.1% (minimize)
    Activity = 91.3% (maximize)
    Stability = 5.9 days (maximize)

... (all Pareto points listed)
```

---

## Limitations and Constraints

### Number of Responses

| # Responses | DoE Analysis | Multi-Objective BO | Pareto Plot |
|-------------|--------------|-------------------|-------------|
| 1 | ✅ Yes | ✅ Single-objective mode | ❌ N/A |
| 2 | ✅ Yes | ✅ Multi-objective mode | ✅ 2D scatter |
| 3 | ✅ Yes | ✅ Multi-objective mode | ✅ 3D scatter |
| 4+ | ✅ Yes | ✅ Multi-objective mode | ❌ Text only |

### Cannot "Join" or Combine Responses

**Question**: "Can I combine Tm and Aggregation into one score?"

**Answer**: No, not in the current implementation.

**Workarounds**:
1. **Add a calculated column to your Excel**:
   ```
   In Excel: Combined_Score = Tm - 2*Aggregation
   Then analyze: Combined_Score (single objective)
   ```

2. **Use Pareto frontier and choose manually**:
   - Let BO find all trade-offs
   - Look at Pareto frontier
   - Choose the point that best matches your priorities

### Cannot Assign Weights

**Question**: "Can I say Tm is 3x more important than Aggregation?"

**Answer**: No, multi-objective BO treats all objectives equally.

**Workaround**: Use a weighted combination as a single objective:
```
In Excel: Weighted_Score = 3*Tm - Aggregation
Then analyze: Weighted_Score (single objective)
```

### Cannot Set Constraints

**Question**: "Can I require Tm > 50°C while optimizing Aggregation?"

**Answer**: Not directly. Constraints must be encoded in your experimental data.

**Workaround**:
1. Filter your Excel to only include rows where Tm > 50°C
2. Run analysis on filtered data
3. BO will only learn from valid experiments

---

## Examples

### Example 1: Two Conflicting Objectives

**Scenario**: Maximize protein stability (Tm) while minimizing aggregation

**Setup**:
```
Responses selected:
  ✓ Tm (Maximize)
  ✓ Aggregation (Minimize)

Model: Auto-selected
```

**Results**:
- **DoE Analysis**:
  - Tm: R² = 0.87, quadratic model selected
  - Aggregation: R² = 0.79, linear model selected

- **Pareto Frontier**: 7 optimal trade-off points identified

- **Interpretation**:
  ```
  Point A: Tm=58°C, Agg=2%  → Best Tm, worst Agg
  Point D: Tm=52°C, Agg=1%  → Balanced
  Point G: Tm=48°C, Agg=0.5% → Best Agg, worst Tm
  ```

### Example 2: Three Objectives with Priorities

**Scenario**: Optimize Tm, Aggregation, and Activity. You care most about Activity.

**Setup**:
```
Responses selected:
  ✓ Tm (Maximize)
  ✓ Aggregation (Minimize)
  ✓ Activity (Maximize)

Model: Auto-selected
```

**Strategy**:
1. Look at **3D Pareto frontier plot**
2. Find points with Activity > 80% (your priority)
3. Among those, pick the one with best Tm/Aggregation trade-off

**Alternative** (if Activity is much more important):
```
Excel: Create column Priority_Score = Activity - 0.5*Aggregation + 0.3*Tm
Analyze: Priority_Score (single objective)
```

### Example 3: More Than 3 Responses

**Scenario**: You measured Tm, Aggregation, Activity, Stability, and Yield (5 responses)

**Setup**:
```
Responses selected: All 5 with appropriate directions

Model: Auto-selected
```

**What Happens**:
- ✅ DoE analyzes all 5 independently
- ✅ Multi-objective BO calculates Pareto frontier
- ❌ No visualization (can't plot 5D)
- ✅ Pareto points shown in Recommendations tab

**Results in Recommendations**:
```
📊 PARETO-OPTIMAL SOLUTIONS

Found 12 Pareto-optimal solutions across 5 objectives.

Solution #1: (Conservative - Low risk)
  pH=7.0, NaCl=50mM, Glycerol=5%, ...
  Tm=52°C, Agg=3%, Activity=75%, Stability=6d, Yield=82%

Solution #5: (Balanced)
  pH=7.5, NaCl=100mM, Glycerol=10%, ...
  Tm=55°C, Agg=5%, Activity=85%, Stability=5d, Yield=75%

Solution #12: (High activity focus)
  pH=8.0, NaCl=150mM, Glycerol=15%, ...
  Tm=51°C, Agg=7%, Activity=95%, Stability=4d, Yield=65%
```

**How to Choose**:
- Read through all Pareto solutions
- Identify your priority (e.g., Activity > 90%)
- Among those, pick based on secondary priorities

---

## Summary

### Key Takeaways

1. **Multi-Response DoE**:
   - Analyzes each response **independently**
   - Each gets its own best-fit model
   - Efficient way to extract insights from all measurements

2. **Multi-Objective BO**:
   - Finds **trade-off solutions** (Pareto frontier)
   - Treats all objectives **equally** (no weighting)
   - Visualizes 2D/3D, text for 4D+

3. **What You Control**:
   - ✅ Which responses to analyze
   - ✅ Maximize or minimize each
   - ✅ Model type (auto or manual)
   - ❌ Response weighting
   - ❌ Response combination
   - ❌ Constraints

4. **Limitations**:
   - Cannot assign importance weights
   - Cannot combine responses (must do in Excel)
   - Cannot set hard constraints (filter data instead)
   - Cannot visualize >3 objectives

### When to Use What

| Your Goal | Recommendation |
|-----------|---------------|
| Understand each response separately | Multi-response DoE (any # of responses) |
| Optimize 1 response | Single-objective mode |
| Find trade-offs between 2-3 responses | Multi-objective BO (see Pareto plot) |
| Optimize 4+ responses | Multi-objective BO (read text solutions) |
| Response is 2x more important | Create weighted score in Excel |
| Must meet minimum requirement | Filter Excel data before import |

---

## Technical Details

### How Pareto Frontier is Extracted

```python
# Ax Platform returns Pareto frontier as:
pareto_frontier = ax_client.get_pareto_optimal_parameters()

# Format: dict of {arm_name: (parameters, values)}
# where values = (mean_dict, cov_dict)

# We extract just the means:
for arm_name, (params, values) in pareto_frontier.items():
    mean_dict = values[0]  # Extract mean values only
    pareto_points.append({
        'parameters': params,  # Factor settings
        'objectives': mean_dict  # Response values (means)
    })
```

### Why No Constraints?

The current implementation uses **Ax Platform's default multi-objective optimization**, which:
- ✅ Finds Pareto frontier
- ✅ Balances exploration vs exploitation
- ❌ Does not support hard constraints (e.g., Tm > 50)

**To add constraints**, would need to:
1. Use Ax's `OutcomeConstraint` class
2. Modify `initialize_optimizer()` to accept constraint specifications
3. Add GUI controls for users to specify constraints

This is a **potential future enhancement**.

### Why Equal Weighting?

Ax Platform's multi-objective BO uses **scalarization** or **Pareto optimization**:
- **Scalarization**: Combines objectives with weights → single objective
- **Pareto optimization**: Finds all non-dominated solutions → multiple solutions

Current implementation uses **Pareto optimization** because:
- ✅ No need to guess weights a priori
- ✅ Shows all trade-offs, user chooses
- ❌ But cannot encode preferences automatically

**To add weighting**, could:
1. Switch to scalarization approach
2. Add "Importance" slider for each response in GUI (1-10)
3. Convert to weighted sum: `score = w1*R1 + w2*R2 + ...`

This is also a **potential future enhancement**.

---

## Frequently Asked Questions

**Q: Can I optimize 10 responses at once?**
A: Technically yes, but not recommended. With 10 objectives, the Pareto frontier becomes huge and hard to interpret. Consider grouping related responses or creating composite scores.

**Q: Why can't I visualize 4+ objectives?**
A: Human perception is limited to 3D. There's no good way to visualize 4D+ data on a 2D screen. Text listing of Pareto points is the best alternative.

**Q: Can the software automatically pick the "best" Pareto point?**
A: No, because "best" depends on your priorities (which we don't know). You must choose based on what matters most to you.

**Q: What if all my Pareto points are bad?**
A: This means your experimental design hasn't found good conditions yet. Use the BO suggestions to run more experiments in promising regions.

**Q: Can I change optimization direction after analysis?**
A: Yes! Re-run the analysis with different directions. The DoE statistics don't change, but the Pareto frontier and BO suggestions will.

**Q: Does multi-objective BO take longer than single-objective?**
A: Slightly, but not significantly. The main computational cost is in the GP model fitting, which scales with number of objectives but is still fast (<10 seconds for typical datasets).

---

*Last updated: 2025-01-18*
