# QUICK REFERENCE: Protein Stability DoE Suite Analysis

## What This System Does
A complete Design of Experiments toolkit for protein stability studies:
- **Designer**: Creates factorial experiment designs with chemical factors
- **Robot Protocol**: Automates buffer preparation on Opentrons robots
- **Analysis**: Performs statistical modeling and suggests next experiments via Bayesian Optimization

## By The Numbers
- **7,688 lines** of Python code
- **14 files** total
- **~800 lines** of duplicated code
- **0 test files** (critical gap)
- **2 monolithic GUI files** (2,254 + 3,413 lines)

## Top 5 Issues to Fix (in priority order)

### 1. ELIMINATE CODE DUPLICATION (HIGH PRIORITY)
**Problem:** Same code exists 2-3 places
- `DoEAnalyzer`: In analysis_tab.py AND core/doe_analyzer.py (different versions!)
- `BayesianOptimizer`: In analysis_tab.py AND core/optimizer.py (full vs skeleton)
- `smart_factor_match()`: Implemented 2x
- `generate_well_position()`: Implemented 3x (!)

**Impact:** Bug fixes must be made multiple times, inconsistent behavior

**Fix:** 15-20 hours of refactoring
```
Create:
  utils/well_mapping.py        - Consolidate well position logic
  utils/constants.py           - Consolidate all constants
  utils/validation.py          - Consolidate input validation
  core/data_handler.py         - Extract from analysis_tab
  core/visualization.py        - Extract plotting classes
```

---

### 2. SEPARATE GUI FROM BUSINESS LOGIC (HIGH PRIORITY)
**Problem:** Business logic buried in GUI files
- `analysis_tab.py` (3,413 lines) should be ~600 lines
- `designer_tab.py` (2,254 lines) should be ~400 lines
- Core modules are orphaned/abandoned

**Impact:** Can't reuse code (CLI, API, batch processing); hard to test

**Fix:** 20-30 hours of restructuring
```
Move these OUT of GUI files:
  DataHandler              → core/data_handler.py
  DoEPlotter              → core/visualization.py
  ResultsExporter         → core/visualization.py
  BayesianOptimizer       → core/optimizer.py (fully)
  FactorModel             → core/factor_model.py
```

---

### 3. ADD UNIT TESTS (CRITICAL)
**Problem:** No tests exist (0% coverage)
- Can't safely refactor
- No quality assurance
- Bugs slip through

**Impact:** Risk of breaking functionality, hard to maintain

**Fix:** 25-40 hours
```
Create tests/:
  test_doe_designer.py         (40 tests)
  test_doe_analyzer.py         (30 tests)
  test_optimizer.py            (25 tests)
  test_well_mapping.py         (30 tests)
  test_project.py              (20 tests)
```
Target: 80%+ coverage of core modules

---

### 4. IMPROVE CODE QUALITY (MEDIUM PRIORITY)
**Problem:** Inconsistent documentation and type hints
- Some modules have full type hints, others have none
- Many functions lack docstrings
- Magic numbers hard-coded throughout

**Impact:** Hard to understand code, IDE help is poor, bugs harder to catch

**Fix:** 15-20 hours
```
Add:
  - Type hints to analysis_tab.py, designer_tab.py
  - Docstrings to all public functions
  - Extract magic numbers to constants
  - Use mypy for type checking
```

---

### 5. FIX DEPENDENCY ISSUES (LOW PRIORITY)
**Problem:** Version constraints too loose, some dependencies missing

**Examples:**
- `openpyxl` missing from requirements-analysis.txt but code uses it
- Version pinning allows breaking changes (e.g., `numpy>=1.20.0` could go to 3.0)
- No Python version constraint specified

**Fix:** 1-2 hours
```
Update requirements files:
  numpy>=1.20.0,<2.0         (was: >=1.20.0)
  pandas>=1.3.0,<3.0         (was: >=1.3.0)
  Add missing openpyxl to requirements-analysis.txt
  Add Python >=3.10 constraint
```

---

## Issue Severity Matrix

```
         EFFORT →
  IMPACT   Low    Medium    High
    ↓
  High     #5     #4        #1, #2
  Medium   
  Low      
```

### What to Tackle First
1. **Start with #1** (Consolidate utilities) - Unblocks other work
2. **Then #2** (Refactor GUI) - Enables #3
3. **Then #3** (Add tests) - Protects all other work
4. **Then #4** (Code quality) - Improves maintainability
5. **Finally #5** (Dependencies) - Easy quick win

---

## Architecture Overview

### Current (BAD)
```
┌─────────────────────────────┐
│  GUI Layer (5.7K lines)     │
│ ┌───────────────────────┐   │
│ │ Designer Tab (2.2K)   │   │
│ │ Analysis Tab (3.4K)   │   │
│ │  - Contains ALL       │   │
│ │    business logic!    │   │
│ └───────────────────────┘   │
└─────────────────────────────┘
            ↓
┌─────────────────────────────┐
│  Core Layer (1.0K lines)    │
│ ┌───────────────────────┐   │
│ │ Abandoned/Orphaned    │   │
│ │ Duplicate code        │   │
│ │ Not used by GUI       │   │
│ └───────────────────────┘   │
└─────────────────────────────┘
```

### Ideal (GOOD)
```
┌──────────────────────────────────┐
│  GUI Layer (1.2K lines)          │
│ ┌────────────────────────────┐   │
│ │ Designer Tab (400-600)     │   │
│ │ Analysis Tab (600-800)     │   │
│ │  - UI ONLY                 │   │
│ └────────────────────────────┘   │
└──────────────────────────────────┘
            ↓
┌──────────────────────────────────┐
│  Core Layer (3.0K+ lines)        │
│ ┌────────────────────────────┐   │
│ │ DoEDesigner               │   │
│ │ DoEAnalyzer               │   │
│ │ BayesianOptimizer         │   │
│ │ DataHandler               │   │
│ │ Visualization             │   │
│ │ FactorModel               │   │
│ │  - All business logic     │   │
│ └────────────────────────────┘   │
└──────────────────────────────────┘
            ↓
┌──────────────────────────────────┐
│  Utils Layer (0.5K lines)        │
│ ┌────────────────────────────┐   │
│ │ well_mapping              │   │
│ │ constants                 │   │
│ │ validation                │   │
│ │ sanitization              │   │
│ └────────────────────────────┘   │
└──────────────────────────────────┘
```

---

## Files Reference

### Critical Files to Understand
| File | Size | Role | Quality |
|------|------|------|---------|
| `analysis_tab.py` | 3.4K | Main analysis GUI | Poor (mixed concerns) |
| `designer_tab.py` | 2.3K | Main design GUI | Poor (mixed concerns) |
| `doe_designer.py` | 242 | Design logic | Good (clean) |
| `doe_analyzer.py` | 197 | Analysis logic | Okay (outdated) |
| `optimizer.py` | 210 | BO logic | Incomplete |
| `project.py` | 200 | Data model | Good |
| `protein_stability_doe.py` | 690 | Robot protocol | Good |
| `sanitization.py` | 152 | Name matching | Good |

### Duplicate Detection
```
Code Location #1              Code Location #2              Lines
─────────────────────────────────────────────────────────────────────
analysis_tab.py (212-560)    core/doe_analyzer.py (11-197)    ~200
analysis_tab.py (821-2110)   core/optimizer.py (17-210)       ~200
designer_tab.py (1190)       core/doe_designer.py (16)        ~50
designer_tab.py (1201)       core/doe_designer.py (41)        ~35
designer_tab.py (1225)       core/doe_designer.py (79)        ~55
designer_tab.py (75-107)     [validation functions] (global)  ~30
```

---

## Recommendations Summary

### MUST DO (breaks things if not done)
1. Consolidate duplicate code - 2-3 places define same logic
2. Add tests before major refactoring - otherwise risk regression

### SHOULD DO (improves code significantly)
3. Separate GUI from business logic - enables reuse
4. Add type hints and docstrings - improves maintainability
5. Centralize constants - prevents sync issues

### NICE TO HAVE (quality improvements)
6. Optimize performance - likely not needed for typical usage
7. Add logging - helpful for debugging
8. Create API documentation - useful for future developers

---

## Effort & Impact Summary

| Task | Effort | Impact | Priority |
|------|--------|--------|----------|
| Consolidate utils | 20 hrs | High | 1 |
| Refactor GUI/Core | 25 hrs | High | 2 |
| Add tests | 30 hrs | Critical | 3 |
| Type hints + docs | 15 hrs | Medium | 4 |
| Dependency cleanup | 2 hrs | Low | 5 |
| **TOTAL** | **92 hrs** | | |

**Bottom Line:** ~2.3 weeks of work transforms codebase from "messy" to "professional"

---

## Next Steps

### Week 1: Foundation
- [ ] Read full CODEBASE_ANALYSIS.md
- [ ] Create utils/constants.py with all shared constants
- [ ] Create utils/well_mapping.py with all position functions
- [ ] Update imports across codebase
- [ ] Add unit tests for utils/

### Week 2-3: Core Refactoring
- [ ] Consolidate DoEAnalyzer in core/
- [ ] Consolidate BayesianOptimizer in core/
- [ ] Extract DataHandler to core/
- [ ] Extract FactorModel to core/
- [ ] Update analysis_tab.py and designer_tab.py to use core versions
- [ ] Add unit tests for core/

### Week 3-4: Quality
- [ ] Add type hints throughout GUI files
- [ ] Add comprehensive docstrings
- [ ] Implement logging
- [ ] Run mypy type checker
- [ ] Update requirements.txt with proper versions

### Then: Polish
- [ ] Code review and cleanup
- [ ] User testing
- [ ] Documentation
- [ ] Performance optimization if needed

