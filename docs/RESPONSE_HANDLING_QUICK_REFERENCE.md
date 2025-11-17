# Response Handling - Quick Reference

## One-Sentence Summary
The system expects a column named **"Response"** with numeric values for regression and Bayesian Optimization.

---

## Critical Code Locations

### File Load Validation
**File:** `gui/tabs/analysis_tab.py`  
**Lines:** 434-440  
**Does:** Validates "Response" column exists at file load

```python
if 'Response' not in self.handler.data.columns:
    raise error("Excel file must have 'Response' column")
```

### Response Column Hardcoding
**File:** `gui/tabs/analysis_tab.py`  
**Line:** 462-463  
**Does:** Hardcodes response column name

```python
response_col = "Response"  # ALWAYS this name
```

### BO Objective Definition
**File:** `core/optimizer.py`  
**Line:** 125  
**Does:** Creates single Ax objective

```python
objectives={self.response_column: ObjectiveProperties(minimize=minimize)}
```

### Historical Data Integration
**File:** `core/optimizer.py`  
**Line:** 149  
**Does:** Attaches response values as trial metrics

```python
self.ax_client.complete_trial(
    trial_index=trial_index,
    raw_data=float(row[self.response_column])
)
```

---

## Data Flow - Response at Each Stage

```
Excel File Load
    ↓
Response column = "Response" (validation at lines 434-440)
    ↓
detect_columns() → response_column = "Response" (line 462)
    ↓
preprocess_data() → drop rows with NaN response (line 100-101)
    ↓
fit_model() → build formula with response as Y (line 106-146)
    ↓
set_data() in BayesianOptimizer → response_column = "Response" (line 49-61)
    ↓
initialize_optimizer() → create Ax objective (line 125)
    ↓
attach_trial() → store response as trial metric (line 149)
    ↓
get_next_suggestions() → generate new experiments (line 154-186)
```

---

## Key Classes and Their Response Handling

### DataHandler
- **stores:** `self.response_column`
- **validates:** Column is numeric, not NaN
- **preprocesses:** Drops rows with NaN in response

### DoEAnalyzer
- **stores:** `self.response_column`
- **uses in:** `build_formula()` - creates `Response ~ Factors`
- **uses in:** `calculate_main_effects()` - groups by factors, aggregates response

### BayesianOptimizer
- **stores:** `self.response_column`
- **uses in:** `initialize_optimizer()` - defines Ax objective
- **uses in:** `get_next_suggestions()` - predicts response values

---

## Response Properties

| Property | Value |
|----------|-------|
| **Column Name** | "Response" (hardcoded) |
| **Data Type** | float |
| **Nullable** | No (NaN values dropped) |
| **Min Rows** | 3-4 (for regression) |
| **Optimization Direction** | maximize (default) or minimize |

---

## Common Issues

### Issue 1: "Missing Response Column" Error
**Cause:** Excel file doesn't have column named "Response"  
**Fix:** Rename your response column to exactly "Response"

### Issue 2: Analysis Fails with Type Error
**Cause:** Response column contains non-numeric values  
**Fix:** Ensure all response values are numbers (no text)

### Issue 3: BO Suggestions Show "NaN"
**Cause:** Model training failed due to insufficient clean data  
**Fix:** Check response column for missing values; need >3 rows

---

## Extension Points for Multi-Response

### To Support Multiple Responses:

1. **Change line 462-463 in analysis_tab.py:**
   ```python
   # Current:
   response_col = "Response"
   
   # Future:
   response_cols = self.handler.detect_response_columns()  # Allow multiple
   ```

2. **Update line 125 in optimizer.py:**
   ```python
   # Current:
   objectives={self.response_column: ObjectiveProperties(minimize=minimize)}
   
   # Future:
   objectives={col: ObjectiveProperties(minimize=minimize_dict[col]) 
              for col in self.response_columns}
   ```

3. **Extend line 149 in optimizer.py:**
   ```python
   # Current:
   raw_data=float(row[self.response_column])
   
   # Future:
   raw_data={col: float(row[col]) for col in self.response_columns}
   ```

---

## Testing Response Handling

### Unit Test Examples

**Location:** `tests/core/test_optimizer.py`

**Test 1: Response Column Validation**
```python
def test_response_column_stored(self):
    optimizer.set_data(..., response_column='Response')
    assert optimizer.response_column == 'Response'
```

**Test 2: Historical Data Attachment**
```python
def test_historical_data_attached():
    # Checks that response values from historical data are stored
    # as trial metrics in Ax experiment
```

---

## Ax-Platform Integration

### Single Objective (Current)
```python
# Ax expects:
objectives = {
    "Response": ObjectiveProperties(minimize=False)  # Maximize
}
```

### Multi-Objective (Future)
```python
# Ax supports:
objectives = {
    "Response_1": ObjectiveProperties(minimize=False),
    "Response_2": ObjectiveProperties(minimize=False),
}
```

Ax will automatically generate Pareto frontier with multiple objectives.

---

## Configuration

### Constants Related to Responses
- **`METADATA_COLUMNS`** in utils/constants.py: Columns excluded from factors (Response not listed, treated separately)
- **`R2_LOW_THRESHOLD`** in core/constants.py: 0.5 - model fit quality
- **No response-specific constants** (hardcoded instead)

---

## Performance Notes

- **Response with <3 unique values:** Regression may fail
- **Response with outliers:** Affects model fit quality, consider transformation
- **Mixed numeric/categorical response:** Not supported (must be numeric)
- **High-dimensional response:** Not applicable (single column only)

---

## Version History

- **v0.4.1+:** Response handling stabilized with pH as ordered categorical
- **All versions:** "Response" column hardcoded requirement

