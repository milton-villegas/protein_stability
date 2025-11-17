# Response Handling - Call Flow & Class Hierarchy

## Complete Call Chain for Response Processing

### From User Perspective (Step-by-Step)

```
User Action: Click "Analyze"
└─> AnalysisTab.analyze_data()
    └─> DataHandler.load_excel(filepath)
        ├─> pd.read_excel(filepath)
        ├─> _load_stock_concentrations()
        └─> self.data = DataFrame
        
    └─> DataHandler.detect_columns(response_col="Response")
        ├─> Identify response_column = "Response"
        ├─> Identify factor_columns = [all except metadata]
        ├─> Classify numeric_factors = [float columns]
        └─> Classify categorical_factors = [object columns]
        
    └─> DataHandler.preprocess_data()
        ├─> Drop metadata columns
        ├─> Drop rows where Response is NaN
        ├─> Fill categorical NaN with 'None'
        ├─> Drop rows with NaN in numeric factors
        └─> return self.clean_data
        
    └─> DoEAnalyzer.set_data(clean_data, ..., response_column="Response")
        └─> Store self.response_column = "Response"
        
    └─> DoEAnalyzer.compare_all_models()
        └─> For each model_type in ['mean', 'linear', 'interactions', ...]:
            ├─> build_formula() → "Response ~ Factor1 + Factor2 + ..."
            ├─> smf.ols(formula, data=self.data).fit()
            └─> Extract coefficients, R², AIC, BIC, predictions, residuals
            
    └─> DoEAnalyzer.select_best_model(comparison_data)
        └─> Score all models using Adj R², BIC, complexity
        
    └─> DoEAnalyzer.fit_model(chosen_model_type)
        ├─> build_formula()
        ├─> statsmodels.OLS.fit()
        └─> return results with Response predictions/residuals
        
    └─> BayesianOptimizer.set_data(clean_data, ..., response_column="Response")
        ├─> self.response_column = "Response"
        └─> _calculate_bounds()
        
    └─> BayesianOptimizer.initialize_optimizer(minimize=False)
        ├─> Build Ax parameters from factors
        ├─> AxClient.create_experiment(
        │   objectives={"Response": ObjectiveProperties(minimize=False)}
        │ )
        │
        └─> For each row in historical data:
            ├─> Build params dict from factors
            ├─> attach_trial(params)
            └─> complete_trial(raw_data=float(row["Response"]))
        
    └─> Display visualizations
        ├─> DoEPlotter.plot_main_effects(response_column="Response")
        ├─> Residuals vs fitted (Response - predictions)
        └─> Interaction plots with Response as y-axis
        
    └─> BayesianOptimizer.get_acquisition_plot()
        ├─> _select_most_important_factors()
        ├─> get_model_predictions_for_parameterizations(
        │   metric_names=["Response"]
        │ ) → [(pred_mean, pred_sem), ...]
        │
        ├─> Calculate Expected Improvement (EI)
        │   EI = (mu - best) * Phi(z) + sigma * phi(z)
        │   where z = (mu - best) / sigma
        │
        └─> Create 4-panel plot:
            ├─> Panel 1: Response Surface (predicted Response)
            ├─> Panel 2: Acquisition Function (Expected Improvement)
            ├─> Panel 3: Model Uncertainty (prediction std error)
            └─> Panel 4: Optimization Progress (cumulative best Response)
            
    └─> BayesianOptimizer.get_next_suggestions(n=5)
        └─> For i in range(n):
            ├─> ax_client.get_next_trial() → (params, trial_idx)
            ├─> Convert sanitized names → original factor names
            ├─> Append to suggestions list
            └─> abandon_trial(trial_idx)
            
    └─> BayesianOptimizer.export_bo_batch_to_files(...)
        ├─> Get BO suggestions
        ├─> Map suggestions to Excel columns
        ├─> Calculate volumes from stock concentrations
        ├─> Append rows to Excel file
        └─> Write CSV for Opentrons
        
User views: Results, plots, and BO suggestions
```

---

## Class Method Call Hierarchy

### AnalysisTab (GUI Orchestration)
```
AnalysisTab
├── browse_file()
│   └─ Opens file dialog
│
├── analyze_data()  ⭐ MAIN ENTRY POINT
│   ├─ handler.load_excel(filepath)
│   ├─ handler.detect_columns("Response")
│   ├─ handler.preprocess_data()
│   ├─ analyzer.set_data(..., response_column="Response")
│   ├─ analyzer.compare_all_models()
│   │  └─ For each model:
│   │     ├─ analyzer.build_formula(model_type)
│   │     └─ analyzer.fit_model(model_type)
│   ├─ analyzer.select_best_model(comparison)
│   ├─ analyzer.fit_model(chosen_model)
│   │  └─ Returns predictions, residuals with response values
│   ├─ plotter.set_data(..., response_column="Response")
│   ├─ display_plots()
│   │  └─ plotter.plot_main_effects()
│   │  └─ plotter.plot_interaction_effects()
│   │  └─ plotter.plot_residuals(predictions, residuals)
│   ├─ display_statistics()
│   ├─ display_recommendations()
│   ├─ optimizer.set_data(..., response_column="Response")
│   ├─ optimizer.initialize_optimizer(minimize=False)
│   ├─ display_optimization_plot()
│   │  └─ optimizer.get_acquisition_plot()
│   └─ Enable export buttons
│
└── export_suggestions()
    └─ optimizer.export_bo_batch_to_files(...)
        └─ Returns (excel_path, csv_path)
```

### DataHandler (Data Preparation)
```
DataHandler
├── __init__()
│   └─ self.response_column = None
│
├── load_excel(filepath)
│   ├─ pd.read_excel(filepath) → self.data
│   └─ _load_stock_concentrations(filepath)
│
├── detect_columns(response_column: str) ⭐ RESPONSE SPECIFIED HERE
│   ├─ self.response_column = response_column
│   ├─ self.factor_columns = [cols except response and metadata]
│   └─ Classify numeric_factors vs categorical_factors
│
└── preprocess_data() → clean_data
    ├─ self.clean_data = self.data.copy()
    ├─ Drop metadata columns
    ├─ Drop rows: self.clean_data.dropna(subset=[self.response_column])
    ├─ Handle categorical NaN
    └─ Drop rows with numeric factor NaN
```

### DoEAnalyzer (Statistical Analysis)
```
DoEAnalyzer
├── __init__()
│   ├─ self.response_column = None
│   └─ self.model = None
│
├── set_data(..., response_column: str)
│   ├─ self.response_column = response_column
│   └─ Store factor info (numeric/categorical)
│
├── build_formula(model_type: str) → formula_string
│   ├─ Start: "Q('{self.response_column}') ~ "
│   ├─ Add terms: factor_terms (main effects)
│   ├─ Add terms: interaction_terms (2-way interactions)
│   ├─ Add terms: squared_terms (quadratic)
│   └─ Return: "Response ~ Factor1 + Factor2 + Factor1:Factor2 + ..."
│
├── fit_model(model_type: str) → results_dict
│   ├─ formula = self.build_formula(model_type)
│   ├─ self.model = smf.ols(formula, data=self.data).fit()
│   ├─ Extract:
│   │  ├─ coefficients (all params + t-values + p-values)
│   │  ├─ model_stats (R², Adj R², RMSE, AIC, BIC)
│   │  ├─ predictions = self.model.fittedvalues
│   │  └─ residuals = self.model.resid
│   └─ Return results_dict
│
├── calculate_main_effects() → dict[factor → effects_df]
│   └─ For each factor:
│       └─ effects = self.data.groupby(factor)[self.response_column].agg(...)
│
└── compare_all_models() → comparison_dict
    └─ Fit all model types, return comparison table with R², BIC, etc.
```

### BayesianOptimizer (Optimization)
```
BayesianOptimizer
├── __init__()
│   ├─ self.ax_client = None
│   ├─ self.response_column = None
│   └─ self.factor_columns = []
│
├── set_data(..., response_column: str)
│   ├─ self.data = data.copy()
│   ├─ self.response_column = response_column ⭐ STORED HERE
│   ├─ Store factor info
│   └─ _calculate_bounds()
│
├── _calculate_bounds()
│   ├─ For numeric factors:
│   │  └─ factor_bounds[factor] = (min, max) from data
│   └─ For categorical factors:
│      └─ factor_bounds[factor] = [unique values]
│
├── initialize_optimizer(minimize: bool = False)
│   ├─ Build parameters list (with sanitized names)
│   ├─ self.ax_client = AxClient()
│   ├─ self.ax_client.create_experiment(
│   │   objectives={self.response_column: ObjectiveProperties(minimize=minimize)} ⭐ KEY LINE
│   │ )
│   │
│   └─ For each historical trial:
│       ├─ Build params dict
│       ├─ self.ax_client.attach_trial(params)
│       └─ self.ax_client.complete_trial(
│           trial_index=trial_idx,
│           raw_data=float(row[self.response_column]) ⭐ RESPONSE VALUE
│         )
│
├── get_next_suggestions(n: int = 5) → list[dict]
│   └─ For i in range(n):
│       ├─ params, trial_idx = self.ax_client.get_next_trial()
│       ├─ Convert sanitized names back to original
│       ├─ Apply rounding/type conversion
│       ├─ suggestions.append(params)
│       └─ self.ax_client.abandon_trial(trial_idx)
│
├── get_acquisition_plot() → matplotlib.Figure
│   ├─ _select_most_important_factors() → [factor_x, factor_y]
│   ├─ Create grid of points in factor space
│   ├─ predictions = self.ax_client.get_model_predictions_for_parameterizations(
│   │   metric_names=[self.response_column] ⭐ SINGLE METRIC
│   │ ) → [(pred_mean, pred_sem) for each point]
│   │
│   ├─ Z_mean = reshape predictions
│   ├─ Z_sem = reshape uncertainties
│   ├─ Calculate EI (Expected Improvement):
│   │  current_best = self.data[self.response_column].max()
│   │  Z_ei = (mu - best) * Phi(z) + sigma * phi(z)
│   │
│   └─ Create 4-panel figure:
│       ├─ Panel 1: contourf(X, Y, Z_mean) - Response Surface
│       ├─ Panel 2: contourf(X, Y, Z_ei) - Acquisition Function
│       ├─ Panel 3: contourf(X, Y, Z_sem) - Model Uncertainty
│       └─ Panel 4: plot(iterations, cumulative_best) - Optimization Progress
│
└── export_bo_batch_to_files(...) → (xlsx_path, csv_path)
    ├─ Get BO suggestions
    ├─ Map to Excel columns
    ├─ Calculate volumes
    ├─ Append to Excel
    └─ Write CSV for robot
```

### DoEPlotter (Visualization)
```
DoEPlotter
├── __init__()
│   └─ self.response_column = None
│
├── set_data(data, factor_columns, response_column)
│   └─ self.response_column = response_column ⭐ STORED HERE
│
├── plot_main_effects(save_path: Optional[str]) → Figure
│   └─ For each factor:
│       ├─ grouped = data.groupby(factor)[self.response_column].agg(['mean', 'std'])
│       ├─ means = [grouped[level, 'mean'] for level in levels]
│       └─ plot(means) with std dev shaded region
│
├── plot_interaction_effects(...) → Figure
│   └─ For each (factor1, factor2) pair:
│       ├─ Diagonal: main effect of factor1
│       ├─ Lower triangle: interaction plot
│       │  └─ For each level of factor2:
│       │     └─ grouped = data[data[factor2]==level2].groupby(factor1)[self.response_column].mean()
│       │     └─ plot(grouped) - one line per factor2 level
│       │
│       └─ Upper triangle: label
│
└── plot_residuals(predictions, residuals) → Figure
    ├─ Panel 1: scatter(predictions, residuals)
    ├─ Panel 2: qqplot (normality of residuals)
    ├─ Panel 3: scatter(predictions, sqrt(abs(standardized_residuals))) - homoscedasticity
    └─ Panel 4: histogram(residuals)
```

---

## Data Structures for Response Handling

### Response Column Storage

```python
# In DataHandler
self.response_column: str = "Response"
self.data: pd.DataFrame  # Contains "Response" column
self.clean_data: pd.DataFrame  # Contains "Response" column (rows filtered)

# In DoEAnalyzer
self.response_column: str = "Response"
self.data: pd.DataFrame  # References clean_data
self.model: statsmodels.OLS  # Model with Response as y-variable
self.results: Dict[str, Any] = {
    'coefficients': pd.DataFrame,
    'model_stats': {
        'R-squared': float,
        'Adjusted R-squared': float,
        'RMSE': float,
        ...
    },
    'predictions': np.ndarray,  # Predicted Response values
    'residuals': np.ndarray  # Response - predictions
}

# In BayesianOptimizer
self.response_column: str = "Response"
self.data: pd.DataFrame  # Includes "Response" column
self.ax_client: AxClient = AxClient()
self.ax_client._experiment.objectives = {
    "Response": ObjectiveProperties(minimize=False)
}
# Each trial in ax_client has:
# - parameters: {factor_name: value, ...}
# - measurements: {metric_name: mean/sem, ...}
#   where metric_name = "Response"
```

### Flow of Response Values

```
Excel File (user data)
├─ Columns: [ID, Factor1, Factor2, ..., Response, ...]
│
└─> pd.read_excel()
    │
    └─> DataFrame.df
        ├─ df["Response"] = [0.5, 0.8, 0.6, ...]
        │
        └─> detect_columns(response_column="Response")
            └─> response_column = "Response"
            
            └─> preprocess_data()
                └─> Drop rows where df["Response"] is NaN
                
                └─> clean_data = df (filtered)
                
                    └─> DoEAnalyzer.set_data(clean_data, ..., response_column="Response")
                        │
                        └─> build_formula()
                            └─> "Response ~ Factor1 + Factor2"
                                └─> smf.ols(formula, data=clean_data).fit()
                                    ├─ Fit OLS regression
                                    ├─ Extract coefficients
                                    ├─ Compute predictions
                                    ├─ Compute residuals = actual - predicted
                                    └─> results['predictions'], results['residuals']
                        
                        └─> BayesianOptimizer.set_data(..., response_column="Response")
                            │
                            └─> initialize_optimizer()
                                └─> For each row in clean_data:
                                    ├─ value = float(row["Response"])
                                    └─> attach_trial() + complete_trial(raw_data=value)
                                        └─> Store in Ax: {metric_name="Response": {mean=value, sem=...}}
```

---

## Response Transformation Pipeline

```
1. RAW INPUT (Excel)
   Response = [0.50, 0.82, 0.58, 0.65, ...]
   
2. DATA LOADING
   Loaded into DataFrame["Response"]
   
3. VALIDATION
   ├─ Check column name = "Response" ✓
   ├─ Check dtype = numeric ✓
   └─ Check not all NaN ✓
   
4. PREPROCESSING
   ├─ Drop rows where Response is NaN
   └─ Result: Response = [0.50, 0.82, 0.58, 0.65, ...]
              (unchanged if no NaN)
   
5. STATISTICAL MODELING
   Formula: "Response ~ Factor1 + Factor2 + ..."
   
   OLS Regression:
   Response_i = intercept + coeff1*Factor1_i + coeff2*Factor2_i + error_i
   
   Predictions: Ŷ = intercept + coeff1*Factor1 + coeff2*Factor2
   Residuals: e = Response - Ŷ
   
6. BAYESIAN OPTIMIZATION
   Historical data → Ax BoTorch model
   
   For each historical trial:
   {parameters: {Factor1: 7.0, Factor2: 100}, 
    observation: {Response: 0.75}}
   
   BO learns: Response ~ f(Factor1, Factor2) [Gaussian Process]
   
7. PREDICTIONS
   For new parameterizations:
   {Factor1: 7.5, Factor2: 150} → GP predicts: {Response: 0.78 ± 0.05}
   
8. ACQUISITION FUNCTION
   EI = (μ - current_best) * Φ(z) + σ * φ(z)
   
   where z = (μ - current_best) / σ
         current_best = max(Response) from history
   
9. VISUALIZATION
   Response values on plot axes and contours
```

---

## Extension Points for Multi-Response

### Code Changes Required

**1. DataHandler (detect_columns)**
```python
# Current:
def detect_columns(self, response_column: str):
    self.response_column = response_column

# Multi-Response Version:
def detect_columns(self, response_columns: List[str]):
    self.response_columns = response_columns
    # Validate all are numeric
```

**2. DoEAnalyzer (fit_model)**
```python
# Current:
def fit_model(self, model_type: str):
    formula = f"Q('{self.response_column}') ~ ..."

# Multi-Response Version:
def fit_model(self, model_type: str):
    # Option A: Separate models per response
    self.models = {
        col: smf.ols(f"Q('{col}') ~ ...", data=self.data).fit()
        for col in self.response_columns
    }
```

**3. BayesianOptimizer (initialize_optimizer)**
```python
# Current:
objectives = {self.response_column: ObjectiveProperties(minimize=minimize)}

# Multi-Response Version:
objectives = {
    col: ObjectiveProperties(minimize=minimize_dict.get(col, False))
    for col in self.response_columns
}
```

**4. BayesianOptimizer (attach_trial)**
```python
# Current:
self.ax_client.complete_trial(
    trial_index=trial_index,
    raw_data=float(row[self.response_column])
)

# Multi-Response Version:
self.ax_client.complete_trial(
    trial_index=trial_index,
    raw_data={
        col: float(row[col])
        for col in self.response_columns
    }
)
```

---

## Summary: Response Handling Layers

| Layer | Component | Responsibility |
|-------|-----------|-----------------|
| **Input** | Excel File | Must have "Response" column |
| **Load** | DataHandler | Validate & load Response column |
| **Preprocess** | DataHandler | Drop NaN rows in Response |
| **Analysis** | DoEAnalyzer | Response as dependent variable |
| **Optimization** | BayesianOptimizer | Response as objective metric |
| **Visualization** | DoEPlotter | Response on plot axes/labels |
| **Export** | Excel/CSV | Response values exported |

