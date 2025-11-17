# Test Coverage Gap Analysis

## Executive Summary

**Current Overall Coverage: ~40-45%**
**Target Coverage: 60%+**
**Missing: 15-20% coverage**

---

## 🔴 CRITICAL GAPS (High Priority)

### 1. **project.py - NO TESTS (189 lines)**

**Impact:** This is your core data model used by both Designer and Analysis tabs.

**Functions with 0% coverage:**
- `__init__()` - Project initialization
- `add_factor()` - Add experimental factors
- `update_factor()` - Update factor levels
- `remove_factor()` - Remove factors
- `get_factors()` - Retrieve factors
- `get_stock_conc()` - Get stock concentrations
- `get_all_stock_concs()` - Get all stock concentrations
- `clear_factors()` - Clear all factors
- `load_results()` - Load Excel results
- `load_stock_concentrations_from_sheet()` - Load stock concentrations
- `detect_columns()` - Detect factor types
- `preprocess_data()` - Clean and prepare data
- `save()` - Save project to file
- `load()` - Load project from file
- `__repr__()` - String representation

**Estimated coverage gain: +8-10%**

**Risk if not tested:**
- ❌ Factor management bugs could corrupt experiments
- ❌ Data loading failures could crash the GUI
- ❌ Save/load bugs could lose user work

---

### 2. **optimizer.py - Partial Coverage (~40%)**

**What we tested (✅):**
- Initialization
- Data handling
- Name sanitization
- Bounds calculation
- Error handling without Ax

**What we MISSED (❌):**
- `get_next_suggestions(n=5)` - **CRITICAL** - Suggests next experiments
- `_create_suggestion_heatmap()` - Creates visualization
- `_select_most_important_factors()` - Feature selection
- `get_acquisition_plot()` - Acquisition function visualization
- `export_bo_plots()` - Export optimization plots
- `_smart_column_match()` - Column name matching
- `export_bo_batch_to_files()` - Export batch suggestions

**Estimated coverage gain: +5-7%**

**Risk if not tested:**
- ❌ Bayesian optimization could suggest invalid experiments
- ❌ Export failures could lose optimization results
- ❌ Plotting errors could crash the analysis tab

---

## ⚠️ MEDIUM GAPS (Medium Priority)

### 3. **Edge Cases in Existing Tests**

Current tests mostly cover "happy path" scenarios. Missing edge cases:

**doe_analyzer.py:**
- Division by zero in statistics
- Empty datasets
- All factors insignificant (pure noise data)
- Multicollinearity (correlated factors)
- Perfect fit (R² = 1.0)
- Underdetermined systems (more params than data)

**optimizer.py:**
- Optimization with no prior data
- Conflicting objectives
- Categorical-only optimization
- Single-factor optimization
- Extreme bounds (very narrow ranges)

**data_handler.py (98% → 100%):**
- Corrupted Excel files
- Missing columns
- Invalid data types
- Unicode characters in column names

**Estimated coverage gain: +3-5%**

---

## ℹ️ LOW PRIORITY GAPS

### 4. **GUI Files (3,893 lines, 0% coverage)**

**Files:**
- `gui/tabs/designer_tab.py` - 2,332 lines
- `gui/tabs/analysis_tab.py` - 1,392 lines
- `gui/main_window.py` - 169 lines

**Why low priority:**
- GUI testing is complex (requires mocking tkinter widgets)
- Visual components are tested manually by users
- Business logic is already separated into core/ modules
- Minimal risk to experimental results

**If you want to test GUI:**
- Use `pytest-qt` or similar frameworks
- Test only critical workflows (file loading, export)
- Focus on integration tests, not unit tests

**Estimated coverage gain: +2-3% (only worth testing critical paths)**

---

## 📊 Coverage Improvement Plan

### Quick Wins (2-3 hours total)

**Step 1: Add project.py tests (1-2 hours)**
→ Coverage: 40% → 48% (+8%)

**Step 2: Complete optimizer.py tests (1 hour)**
→ Coverage: 48% → 53% (+5%)

**Step 3: Add edge case tests (30 min)**
→ Coverage: 53% → 56% (+3%)

**Result: 56% coverage = Above 50% threshold!**

---

### Stretch Goals (Additional 2-3 hours)

**Step 4: More edge cases (+2%)**
**Step 5: Integration tests (+2%)**
**Step 6: Critical GUI paths (+2%)**

**Result: 62% coverage = Professional grade!**

---

## 🎯 Recommended Action Plan

### TODAY: Fix Critical Gaps

1. **Create `tests/core/test_project.py`** (highest ROI)
   - Test factor management (add, update, remove, get)
   - Test save/load functionality
   - Test data loading and preprocessing
   - Test column detection

2. **Enhance `tests/core/test_optimizer.py`**
   - Add tests for `get_next_suggestions()`
   - Add tests for export functions
   - Add tests for plotting (mock matplotlib)

### THIS WEEK: Add Edge Cases

3. **Add edge case tests to existing files**
   - Empty data
   - Invalid inputs
   - Extreme values
   - Error conditions

---

## 📋 Detailed Test Examples

### Priority 1: project.py Tests

```python
# tests/core/test_project.py

def test_add_factor():
    """Test adding a factor"""
    project = DoEProject()
    project.add_factor("pH", ["7.0", "7.5", "8.0"])

    factors = project.get_factors()
    assert "pH" in factors
    assert len(factors["pH"]) == 3

def test_add_factor_empty_name_raises_error():
    """Test that empty factor name raises error"""
    project = DoEProject()
    with pytest.raises(ValueError, match="cannot be empty"):
        project.add_factor("", ["7.0"])

def test_save_and_load(tmp_path):
    """Test project save/load round-trip"""
    project = DoEProject()
    project.name = "Test Project"
    project.add_factor("NaCl", ["100", "200"])

    # Save
    save_path = tmp_path / "project.pkl"
    project.save(str(save_path))

    # Load
    loaded = DoEProject.load(str(save_path))
    assert loaded.name == "Test Project"
    assert "NaCl" in loaded.get_factors()
```

### Priority 2: optimizer.py Additional Tests

```python
# Add to tests/core/test_optimizer.py

@pytest.mark.skipif(not AX_AVAILABLE, reason="Requires ax-platform")
def test_get_next_suggestions(simple_optimization_data):
    """Test generating next experiment suggestions"""
    optimizer = BayesianOptimizer()
    optimizer.set_data(...)
    optimizer.initialize_optimizer()

    # Get suggestions
    suggestions = optimizer.get_next_suggestions(n=3)

    assert len(suggestions) == 3
    assert all('pH' in s for s in suggestions)
    assert all('NaCl' in s for s in suggestions)

def test_export_bo_plots(tmp_path):
    """Test exporting Bayesian optimization plots"""
    optimizer = BayesianOptimizer()
    # ... set up optimizer ...

    export_dir = tmp_path / "plots"
    export_dir.mkdir()

    optimizer.export_bo_plots(
        directory=str(export_dir),
        base_name="TestExperiment"
    )

    # Check files were created
    assert (export_dir / "TestExperiment_bo_plots.png").exists()
```

### Priority 3: Edge Case Tests

```python
# Add to existing test files

def test_analyzer_with_empty_dataframe():
    """Test analyzer handles empty data gracefully"""
    analyzer = DoEAnalyzer()
    empty_df = pd.DataFrame()

    analyzer.set_data(empty_df, [], [], [], "Response")

    with pytest.raises(ValueError, match="No data|Insufficient data"):
        analyzer.fit_model('linear')

def test_optimizer_with_single_factor():
    """Test optimization with only one factor"""
    data = pd.DataFrame({
        'pH': [7.0, 7.5, 8.0],
        'Response': [0.5, 0.8, 0.6]
    })

    optimizer = BayesianOptimizer()
    optimizer.set_data(data, ['pH'], [], ['pH'], 'Response')

    # Should handle single factor without errors
    optimizer.initialize_optimizer()
```

---

## 🔧 Tools to Identify Missing Coverage

### 1. Generate Coverage Report

```bash
pytest tests/ --cov=core --cov-report=html --cov-report=term-missing
```

Output shows missing lines:
```
core/project.py          0%   18-189
core/optimizer.py       42%   154-188, 266-310, 411-669, 670-900
```

### 2. View Interactive HTML Report

```bash
open htmlcov/index.html
```

Click on any file to see:
- ✅ Green lines = tested
- ❌ Red lines = not tested
- ⚠️ Yellow lines = partially tested

### 3. Focus Testing Efforts

```bash
# Test only project.py coverage
pytest tests/core/test_project.py --cov=core.project --cov-report=term-missing

# Test only optimizer.py coverage
pytest tests/core/test_optimizer.py --cov=core.optimizer --cov-report=term-missing
```

---

## 📈 Expected Results

### Current State
```
Overall:        40-45%
project.py:     0%     ← CRITICAL GAP
optimizer.py:   42%    ← PARTIAL GAP
doe_analyzer:   75%    ✓ Good
data_handler:   98%    ✓ Excellent
plotter:        99%    ✓ Excellent
exporter:       100%   ✓ Perfect
```

### After Implementing This Plan
```
Overall:        60-62%  ← TARGET ACHIEVED!
project.py:     85%     ✓ Well tested
optimizer.py:   70%     ✓ Good coverage
doe_analyzer:   80%     ✓ Improved
data_handler:   100%    ✓ Perfect
plotter:        100%    ✓ Perfect
exporter:       100%    ✓ Perfect
```

---

## 🎓 Summary

**Missing tests for:**
1. ❌ project.py (189 lines) - **Most critical**
2. ⚠️ optimizer.py - Missing 7 key functions
3. ⚠️ Edge cases across all modules
4. ℹ️ GUI files (low priority)

**Recommended focus:**
1. Write project.py tests first (+8% coverage)
2. Complete optimizer.py tests (+5% coverage)
3. Add edge cases (+3% coverage)
4. **Result: 56-60% coverage achieved!**

**Time investment:**
- 2-3 hours to reach 56%
- 4-6 hours to reach 62%
- Great ROI for code quality!

