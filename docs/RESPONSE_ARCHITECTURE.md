# Response Handling Architecture - Protein Stability DoE Toolkit

## Executive Summary

The protein stability DoE toolkit currently uses a **single-response architecture** where:
- A single "Response" column is designated for optimization
- Bayesian Optimization (via Ax-Platform) handles single-objective optimization
- Response handling is tightly coupled to the GUI and analysis workflow
- No built-in support for multi-response or multi-objective optimization

---

## 1. RESPONSE VARIABLE DEFINITION AND USAGE

### 1.1 Response Column Specification

**Current Implementation:**
- Response column is **hardcoded as "Response"** in the GUI (analysis_tab.py:462-463)
- Must be named exactly "Response" in the Excel file
- File loading fails with error if "Response" column is missing (analysis_tab.py:434-440)

```python
# From gui/tabs/analysis_tab.py:462-463
response_col = "Response"  # Always hardcoded

# From gui/tabs/analysis_tab.py:434-440
if 'Response' not in self.handler.data.columns:
    messagebox.showerror("Missing Response Column",
                       "Excel file must have a column named 'Response'\n\n"
                       "Please rename your response column to 'Response' and try again.")
```

### 1.2 Response Column Properties

**Data Type:** Numeric (float)
- Must be numeric for regression and Bayesian Optimization
- Non-numeric values cause analysis to fail
- NaN/missing values are removed during preprocessing

**Preprocessing:**
```python
# From core/data_handler.py:100-101
self.clean_data = self.clean_data.dropna(subset=[self.response_column])
```

### 1.3 Response Usage Throughout the System

| Module | Usage | Purpose |
|--------|-------|---------|
| **data_handler.py** | `self.response_column` | Data loading/preprocessing |
| **doe_analyzer.py** | `self.response_column` | Formula building for regression |
| **optimizer.py** | `self.response_column` | BO objective definition |
| **plotter.py** | `self.response_column` | Visualization (axes labels) |
| **analysis_tab.py** | `self.response_column` | GUI orchestration |

### 1.4 Response Role in Different Workflows

**Statistical Analysis:**
```python
# From core/doe_analyzer.py:106-146
formula = f"Q('{self.response_column}') ~ " + " + ".join(factor_terms)
# Builds regression formula: Response ~ Factor1 + Factor2 + ...
```

**Main Effects Calculation:**
```python
# From core/doe_analyzer.py:233-237
effects = self.data.groupby(factor)[self.response_column].agg(['mean', 'std', 'count'])
# Groups by factor levels and computes response statistics
```

---

## 2. BAYESIAN OPTIMIZATION IMPLEMENTATION

### 2.1 BayesianOptimizer Class Architecture

**Location:** `/home/user/protein_stability/core/optimizer.py`

**Key Components:**
```python
class BayesianOptimizer:
    def __init__(self):
        self.ax_client = None              # Ax AxClient instance
        self.response_column = None        # Single response target
        self.factor_columns = []           # List of experimental factors
        self.numeric_factors = []          # Numeric/continuous factors
        self.categorical_factors = []      # Categorical factors
        self.factor_bounds = {}            # Min/max for numeric, values for categorical
        self.is_initialized = False
        self.name_mapping = {}             # Sanitized → original factor names
        self.reverse_mapping = {}          # Original → sanitized factor names
```

### 2.2 Initialization Pipeline

**Step 1: Data Setup**
```python
# optimizer.py:49-61
def set_data(self, data, factor_columns, categorical_factors, 
             numeric_factors, response_column):
    self.data = data.copy()
    self.factor_columns = factor_columns
    self.categorical_factors = categorical_factors
    self.numeric_factors = numeric_factors
    self.response_column = response_column
    self._calculate_bounds()
```

**Step 2: Factor Bounds Calculation**
```python
# optimizer.py:63-74
def _calculate_bounds(self):
    # Numeric factors: min/max from data
    for factor in self.numeric_factors:
        min_val = float(self.data[factor].min())
        max_val = float(self.data[factor].max())
        self.factor_bounds[factor] = (min_val, max_val)
    
    # Categorical factors: unique values
    for factor in self.categorical_factors:
        unique_vals = self.data[factor].unique().tolist()
        self.factor_bounds[factor] = unique_vals
```

**Step 3: Ax Experiment Creation**
```python
# optimizer.py:121-127
self.ax_client = AxClient()
self.ax_client.create_experiment(
    name="doe_optimization",
    parameters=parameters,
    objectives={self.response_column: ObjectiveProperties(minimize=minimize)},
    choose_generation_strategy_kwargs={"num_initialization_trials": 0}
)
```

**Key Point:** Single objective is created from `self.response_column`

### 2.3 Objective Handling

**Current Single-Objective Setup:**
```python
# optimizer.py:125
objectives={self.response_column: ObjectiveProperties(minimize=minimize)}
```

**Constraint:**
- Only one key-value pair in objectives dict
- `minimize=False` for maximization (default for protein stability experiments)
- `minimize=True` optional for minimization problems

### 2.4 Data Integration with Existing Results

Historical experimental data is attached as completed trials:
```python
# optimizer.py:130-150
for idx, row in self.data.iterrows():
    # Build parameter dict with sanitized names
    params = {}
    for factor in self.factor_columns:
        params[sanitized_name] = value  # Type-converted
    
    # Attach trial to Ax experiment
    _, trial_index = self.ax_client.attach_trial(parameters=params)
    self.ax_client.complete_trial(
        trial_index=trial_index,
        raw_data=float(row[self.response_column])  # Single response value
    )
```

### 2.5 Suggestion Generation

**Getting Next Suggestions:**
```python
# optimizer.py:154-186
def get_next_suggestions(self, n=5):
    suggestions = []
    for _ in range(n):
        params, trial_index = self.ax_client.get_next_trial()
        
        # Convert sanitized names back to original display names
        original_params = {}
        for sanitized_name, value in params.items():
            original_name = self.name_mapping[sanitized_name]
            # Apply rounding/type conversion
            original_params[original_name] = value
        
        suggestions.append(original_params)
        self.ax_client.abandon_trial(trial_index)  # Don't lock suggestions
    
    return suggestions
```

**Return Format:** List of dicts with factor names as keys

### 2.6 Model Predictions and Visualization

**Batch Prediction:**
```python
# optimizer.py:480-495
predictions_list = self.ax_client.get_model_predictions_for_parameterizations(
    parameterizations=parameterizations,
    metric_names=[self.response_column]  # Single metric
)

# Extract predictions and uncertainty
for idx in range(len(predictions_list)):
    pred_mean, pred_sem = predictions_list[idx][self.response_column]
    Z_mean[i, j] = pred_mean
    Z_sem[i, j] = pred_sem
```

**Acquisition Function (Expected Improvement):**
```python
# optimizer.py:543-555
current_best = self.data[self.response_column].max()

# Proper EI formula
Z_score = (Z_mean - current_best) / Z_sem_safe
Z_ei = (Z_mean - current_best) * scipy_stats.norm.cdf(Z_score) + \
       Z_sem_safe * scipy_stats.norm.pdf(Z_score)
Z_ei = np.maximum(Z_ei, 0)
```

### 2.7 BO Export Workflow

**Batch Export to Excel & CSV:**
```python
# optimizer.py:952-1307
def export_bo_batch_to_files(self, n_suggestions, batch_number, excel_path,
                            stock_concs, final_volume, buffer_ph_values):
    # Generates n_suggestions BO recommendations
    # Appends to existing Excel file
    # Creates Opentrons-compatible CSV with calculated volumes
```

**Key Operations:**
1. Generate BO suggestions
2. Map suggestions to factor columns in Excel
3. Calculate liquid handling volumes from stock concentrations
4. Export both XLSX (for documentation) and CSV (for robot)

---

## 3. OPTIMIZATION TARGETS/OBJECTIVES HANDLING

### 3.1 Current Single-Objective Approach

**Objective Definition:**
```python
# optimizer.py:49-55
self.response_column = None  # Single target metric

# optimizer.py:125
ObjectiveProperties(minimize=minimize)  # minimize=False for maximization
```

**Limitation:** Only supports one optimization direction
- No Pareto frontier exploration
- No weighted multi-objective optimization
- No constraint handling (soft or hard)

### 3.2 Optimization Direction

**Maximize (Default):**
```python
# optimizer.py:76
def initialize_optimizer(self, minimize=False):
    # minimize=False → maximization (protein stability typically maximized)
```

**Example:** If Response = Thermal Stability (TM), maximize TM

**Minimize (Optional):**
```python
optimizer.initialize_optimizer(minimize=True)
# For objectives where lower is better (e.g., aggregation rate)
```

### 3.3 Response Selection UI

**Static Requirement (analysis_tab.py):**
- Response column must be named exactly "Response"
- No dropdown or user selection for response column name
- Fails explicitly if column not found

```python
# analysis_tab.py:434-440
if 'Response' not in self.handler.data.columns:
    messagebox.showerror("Missing Response Column",
                       "Excel file must have a column named 'Response'...")
```

### 3.4 BO Objective Metrics in GUI

**Display:**
- Model R² shown in status bar
- Best response value shown in recommendations
- Optimization progress tracked (cumulative best)

```python
# analysis_tab.py:807-813
max_idx = clean_data[self.handler.response_column].idxmax()
optimal_response = clean_data.loc[max_idx, self.handler.response_column]
self.recommendations_text.insert(tk.END, f"Response Value: {optimal_response:.2f}\n\n")
```

---

## 4. OVERALL RESPONSE HANDLING ARCHITECTURE

### 4.1 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ ANALYSIS TAB (analysis_tab.py)                              │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ 1. Browse Excel File                                    │ │
│ │    ↓                                                    │ │
│ │ 2. Load Data (DataHandler.load_excel)                   │ │
│ │    ├─ Check for "Response" column [HARDCODED]           │ │
│ │    └─ Load stock concentrations from metadata sheet     │ │
│ │    ↓                                                    │ │
│ │ 3. Detect Columns (DataHandler.detect_columns)          │ │
│ │    ├─ response_column = "Response"                      │ │
│ │    ├─ factor_columns = all others except metadata       │ │
│ │    ├─ numeric_factors vs categorical_factors detection  │ │
│ │    ↓                                                    │ │
│ │ 4. Preprocess (DataHandler.preprocess_data)             │ │
│ │    ├─ Drop metadata columns                             │ │
│ │    ├─ Drop rows with missing Response                   │ │
│ │    ├─ Fill categorical NaN with 'None'                  │ │
│ │    └─ Drop rows with missing numeric factors            │ │
│ │    ↓                                                    │ │
│ │ 5. Statistical Analysis (DoEAnalyzer)                   │ │
│ │    ├─ Build regression formula with Response            │ │
│ │    ├─ Fit multiple model types                          │ │
│ │    ├─ Calculate R², Adj R², BIC for model selection     │ │
│ │    └─ Generate predictions and residuals                │ │
│ │    ↓                                                    │ │
│ │ 6. Bayesian Optimization (BayesianOptimizer)            │ │
│ │    ├─ set_data(response_column="Response")              │ │
│ │    ├─ initialize_optimizer(minimize=False)              │ │
│ │    │  └─ Create Ax experiment with single objective     │ │
│ │    ├─ Attach historical data as completed trials        │ │
│ │    ├─ get_next_suggestions(n=5)                         │ │
│ │    ├─ Visualize BO with acquisition plots               │ │
│ │    └─ export_bo_batch_to_files()                        │ │
│ │    ↓                                                    │ │
│ │ 7. Export Results                                        │ │
│ │    ├─ Excel with model summary and BO suggestions       │ │
│ │    ├─ CSV for Opentrons robot                           │ │
│ │    └─ High-resolution BO visualization plots            │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ CORE MODULES                                                │
├──────────────────┬──────────────────┬──────────────────────┤
│ DataHandler      │ DoEAnalyzer      │ BayesianOptimizer  │
├──────────────────┼──────────────────┼──────────────────────┤
│ • load_excel()   │ • fit_model()    │ • set_data()       │
│ • detect_cols()  │ • compare_models │ • initialize()     │
│ • preprocess()   │ • calculate_me() │ • get_suggestions()│
└──────────────────┴──────────────────┴──────────────────────┘
```

### 4.2 Response Column Integration Points

| Component | File | Line(s) | Action |
|-----------|------|---------|--------|
| **GUI Load** | analysis_tab.py | 434-440 | Validate "Response" column exists |
| **Data Detection** | data_handler.py | 64-85 | Identify response vs factors |
| **Preprocessing** | data_handler.py | 100-101 | Drop rows with missing response |
| **Formula Building** | doe_analyzer.py | 106-146 | Create regression formula |
| **Model Fitting** | doe_analyzer.py | 163-166 | Fit OLS with response as y-variable |
| **Main Effects** | doe_analyzer.py | 235-236 | Group by factors, aggregate response |
| **BO Setup** | optimizer.py | 49-61 | Store response column name |
| **BO Objective** | optimizer.py | 125 | Create Ax objective from response |
| **BO Training** | optimizer.py | 149 | Attach response values as trial metrics |
| **BO Prediction** | optimizer.py | 480-495 | Get predicted response and uncertainty |
| **Visualization** | plotter.py | 44-55 | Use response for plot labels/axes |

### 4.3 Response Data Type and Validation

**Type Requirements:**
- Float numeric type
- Positive or negative values acceptable
- No categorical/string values

**Validation Chain:**
1. Excel file must be loadable
2. Column named "Response" must exist
3. Values must be numeric (or convertible to float)
4. At least 3+ rows required for regression

**Error Handling:**
```python
# DataFrame.astype(float) will raise if conversion fails
# NaN/None values filtered during preprocessing
# Empty factor columns cause fit_model to fail
```

### 4.4 Response in Project Persistence

**Storage (project.py):**
```python
# core/project.py:31
self.response_column: Optional[str] = None  # Stored in DoEProject
```

**Persistence:**
- Saved in pickle format (.doe file)
- Restored on project load
- Used to regenerate BO state

---

## 5. MULTI-RESPONSE HANDLING (CURRENT STATE)

### 5.1 Existing Limitations

**No Support For:**
- Multiple response columns (e.g., TM and Aggregation)
- Multi-objective optimization (Pareto frontier)
- Weighted response combinations
- Constraint satisfaction
- Priority/hierarchical objectives

### 5.2 Current Single-Response Bottlenecks

**Hardcoded "Response" Column:**
```python
# analysis_tab.py:462-463
response_col = "Response"  # HARDCODED - prevents multiple responses
```

**Single Objective in Ax:**
```python
# optimizer.py:125
objectives={self.response_column: ObjectiveProperties(minimize=minimize)}
# Dict with single key - Ax supports multiple keys but code doesn't use them
```

**GUI Assumptions:**
- Single response selected implicitly
- No multi-objective selection widget
- No weighting/priority interface

### 5.3 Points of Extension for Multi-Response

**Potential Modifications Needed:**

1. **Response Selection** (analysis_tab.py)
   - Add dropdown to select multiple response columns
   - Support weighted combination vs Pareto frontier mode

2. **Data Handler** (data_handler.py)
   - Generalize from single `response_column` to `response_columns` (list)
   - Handle multiple numeric columns with NaN

3. **DoE Analyzer** (doe_analyzer.py)
   - Build separate models for each response
   - Or use multivariate regression for joint optimization

4. **Bayesian Optimizer** (optimizer.py)
   - Create multiple objectives in Ax
   - Handle Pareto frontier generation
   - Weight objectives or use lexicographic optimization

5. **Exporter** (exporter.py)
   - Track multiple BO objectives
   - Export separate columns for each response prediction

---

## 6. CONFIGURATION FILES

### 6.1 Design Configuration
**File:** `/home/user/protein_stability/config/design_config.py`

- Defines factor constraints (min/max/type)
- Unit options for display
- Design type specifications
- NOT directly involved in response handling

### 6.2 Core Constants
**File:** `/home/user/protein_stability/core/constants.py`

- Statistical thresholds (α=0.05)
- Model selection weights (Adj R² 60%, BIC 30%, complexity 10%)
- Plotting parameters (DPI, colors, fonts)
- NO response-related constants

### 6.3 Shared Constants
**File:** `/home/user/protein_stability/utils/constants.py`

```python
METADATA_COLUMNS = ['ID', 'Plate_96', 'Well_96', 'Well_384', 'Source', 'Batch']
```
- Columns excluded from factor detection
- Response column not listed (implicit separate category)

---

## 7. ARCHITECTURE SUMMARY TABLE

| Aspect | Current Implementation | Limitations |
|--------|------------------------|-------------|
| **Response Definition** | Single "Response" column (hardcoded name) | No flexibility, no multi-response |
| **Response Type** | Numeric float | No mixed types or categorical responses |
| **Response Direction** | Maximize or minimize (binary choice) | No constraints or soft bounds |
| **BO Objective** | Single Ax objective | Can't handle Pareto optimization |
| **Model Type** | Univariate regression | Can't model response correlations |
| **Export Format** | XLSX + CSV for single metric | No separate columns per response |
| **UI Selection** | Implicit (must be "Response") | No dynamic selection |
| **Validation** | Checked at file load | Fails early if missing |

---

## 8. KEY FILES AND CODE LOCATIONS

### Core Implementation Files
- **Optimizer:** `/home/user/protein_stability/core/optimizer.py` (1309 lines)
- **Analyzer:** `/home/user/protein_stability/core/doe_analyzer.py` (454 lines)
- **Data Handler:** `/home/user/protein_stability/core/data_handler.py` (113 lines)
- **Project Model:** `/home/user/protein_stability/core/project.py` (190 lines)

### GUI Integration
- **Analysis Tab:** `/home/user/protein_stability/gui/tabs/analysis_tab.py` (1456 lines)
  - Response file validation: lines 434-440
  - Response column hardcoding: line 462-463
  - BO workflow: lines 700-900+

### Configuration
- **Design Config:** `/home/user/protein_stability/config/design_config.py`
- **Core Constants:** `/home/user/protein_stability/core/constants.py`
- **Shared Constants:** `/home/user/protein_stability/utils/constants.py`

### Tests
- **BO Tests:** `/home/user/protein_stability/tests/core/test_optimizer.py`
- **Analyzer Tests:** `/home/user/protein_stability/tests/core/test_doe_analyzer.py`
- **Data Handler Tests:** `/home/user/protein_stability/tests/core/test_data_handler.py`

---

## 9. DEPENDENCIES AND EXTERNAL LIBRARIES

### Statistical Analysis
- **statsmodels** - OLS regression, formula building
- **numpy** - Numerical computations
- **pandas** - DataFrame operations

### Bayesian Optimization
- **Ax-Platform** (`ax.service.ax_client`)
  - AxClient for experiment management
  - ObjectiveProperties for objective definition
  - BoTorch for GP modeling

### Visualization
- **matplotlib** - 2D plotting, contours, heatmaps
- **seaborn** - Style and color palettes
- **scipy.stats** - Statistical distributions (CDF, PDF for EI)

---

## 10. WORKFLOW SUMMARY

### Standard Analysis Workflow

1. **Load Data**: Excel file with factor columns and "Response" column
2. **Detect Structure**: Identify factors (numeric/categorical) from columns
3. **Preprocess**: Remove metadata, handle missing values
4. **Analyze**: Fit regression models with Response as dependent variable
5. **Select Model**: Choose best based on R², BIC, parsimony
6. **Optimize**: Use Bayesian Optimization to suggest next experiments
7. **Export**: Generate XLSX with BO suggestions + CSV for robot

### Response Role at Each Stage

| Stage | Response Usage | Input | Output |
|-------|----------------| ------|--------|
| Load | Validate presence | Excel file | Loaded DataFrame |
| Detect | Identify as target | Column names | response_column = "Response" |
| Preprocess | Drop NaN rows | Raw DataFrame | Clean DataFrame |
| Analyze | Build formula Y ~ X | Clean data | Model coefficients |
| Select | Calculate R² | All models | Best model choice |
| Optimize | Define objective | Historical data | BO suggestions |
| Export | Append to Excel | BO results | New experimental batches |

---

## CONCLUSION

The protein stability DoE toolkit implements a **clean, single-response optimization architecture** with:

✓ **Strengths:**
- Clear data flow from load → analyze → optimize → export
- Tight Ax-Platform integration for modern BO
- Statistical rigor with model comparison
- Publication-quality visualizations

✗ **Limitations for Multi-Response:**
- Hardcoded "Response" column name prevents flexibility
- Single Ax objective prevents Pareto frontier exploration
- No UI support for response selection or weighting
- Would require significant refactoring for multi-objective support

**Recommended Approach for Multi-Response:**
1. Generalize response column specification in UI
2. Create response strategy interface (single vs multi-objective)
3. Extend Ax usage to define multiple objectives with weights
4. Implement Pareto frontier visualization and export

