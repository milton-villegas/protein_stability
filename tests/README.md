# Test Suite Documentation

## Overview

This directory contains the unit test suite for the Protein Stability DoE Suite. The tests validate core business logic modules that were refactored from the GUI (Priority 2).

**Current Coverage:** 21% (71 tests passing)

## Test Structure

```
tests/
├── README.md                 # This file
├── conftest.py              # Shared fixtures and test data
└── core/                    # Core module tests
    ├── test_data_handler.py    # Data loading & preprocessing (17 tests)
    ├── test_exporter.py        # Results export to Excel (14 tests)
    ├── test_plotter.py         # DoE visualization (14 tests)
    └── test_doe_analyzer.py    # Statistical analysis (26 tests)
```

## Running Tests

### Install Dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Run All Tests

```bash
pytest tests/
```

### Run Specific Test File

```bash
pytest tests/core/test_data_handler.py -v
```

### Run with Coverage Report

```bash
pytest tests/ --cov=core --cov-report=html
# Open htmlcov/index.html in browser
```

### Run Specific Test Class or Function

```bash
# Run single test class
pytest tests/core/test_data_handler.py::TestDataHandlerInit -v

# Run single test function
pytest tests/core/test_data_handler.py::TestDataHandlerInit::test_init_creates_empty_attributes -v
```

## What's Tested

### 1. DataHandler (`test_data_handler.py`) - 98% Coverage

**Purpose:** Validates data loading, preprocessing, and stock concentration handling.

**Key Tests:**
- Loading Excel files with experiment data
- Detecting factor types (categorical vs numeric)
- Handling missing values (dropna for numeric, fillna for categorical)
- Loading stock concentrations from metadata sheets
- Removing metadata columns (ID, Plate_96, Well_96, etc.)

**Example:**
```python
def test_load_excel_success(temp_excel_file):
    handler = DataHandler()
    handler.load_excel(str(temp_excel_file))

    assert handler.data is not None
    assert len(handler.data) == 5
```

### 2. ResultsExporter (`test_exporter.py`) - 100% Coverage

**Purpose:** Validates export of statistical results to Excel format.

**Key Tests:**
- Creating multi-sheet Excel files (Model Statistics, Coefficients, Main Effects, Significant Factors)
- Proper formatting of p-values in scientific notation
- Filtering significant factors (p < 0.05)
- Handling DataFrame export/import round-trips

**Example:**
```python
def test_export_creates_file(sample_results, tmp_path):
    exporter = ResultsExporter()
    exporter.set_results(sample_results, sample_main_effects)

    filepath = tmp_path / "test_export.xlsx"
    exporter.export_statistics_excel(str(filepath))

    assert filepath.exists()
```

### 3. DoEPlotter (`test_plotter.py`) - 99% Coverage

**Purpose:** Validates creation of publication-quality DoE plots.

**Key Tests:**
- Main effects plots (mean ± std dev per factor level)
- Interaction plots (factor interaction matrix)
- Residual diagnostic plots (4-panel: residuals vs fitted, Q-Q, scale-location, histogram)
- Saving plots to various formats (PNG, PDF, TIFF, EPS)
- Colorblind-safe color palette usage

**Example:**
```python
def test_plot_main_effects_creates_figure(sample_plot_data):
    plotter = DoEPlotter()
    plotter.set_data(sample_plot_data, ['Factor1', 'Factor2'], 'Response')

    fig = plotter.plot_main_effects()

    assert fig is not None
    assert isinstance(fig, plt.Figure)
```

### 4. DoEAnalyzer (`test_doe_analyzer.py`) - 49% Coverage

**Purpose:** Validates statistical regression models and DoE analysis.

**Key Tests:**
- Formula building for different model types (mean, linear, interactions, quadratic)
- Regression model fitting using statsmodels
- Extracting coefficients, p-values, R², AIC/BIC
- Identifying significant factors
- Calculating main effects (mean/std/count per level)
- Predicting responses for new factor combinations
- Handling categorical and numeric factors

**Example:**
```python
def test_fit_model_linear(simple_2factor_data):
    analyzer = DoEAnalyzer()
    analyzer.set_data(
        data=simple_2factor_data,
        factor_columns=['Factor1', 'Factor2'],
        categorical_factors=[],
        numeric_factors=['Factor1', 'Factor2'],
        response_column='Response'
    )

    results = analyzer.fit_model('linear')

    assert results['model_stats']['R-squared'] > 0.8
```

## Test Fixtures

Located in `conftest.py`, these provide reusable test data:

- `sample_doe_data` - Simple 2-factor experiment with 5 runs
- `sample_doe_data_with_missing` - Data with NaN values for testing preprocessing
- `temp_excel_file` - Temporary Excel file for I/O testing
- `temp_excel_with_stock_concs` - Excel with Stock_Concentrations metadata sheet
- `sample_factorial_design` - 2² factorial design
- `sample_analysis_results` - Pre-computed analysis results for export testing

## Test Organization

Each test file follows this structure:

1. **Initialization Tests** - Verify object creation and default values
2. **Core Functionality Tests** - Test main methods and workflows
3. **Error Handling Tests** - Verify proper exceptions for invalid inputs
4. **Integration Tests** - Test complete workflows end-to-end

## Coverage Goals

| Module | Current | Target |
|--------|---------|--------|
| DataHandler | 98% | 100% |
| ResultsExporter | 100% | 100% ✓ |
| DoEPlotter | 99% | 100% |
| DoEAnalyzer | 49% | 60% |

**Not Tested (intentionally):**
- GUI code (`gui/tabs/*.py`) - Complex GUI testing, low ROI
- BayesianOptimizer - Non-deterministic, optional dependency
- DoEDesigner - Complex experimental design generation

## Running Tests in CI/CD

Tests are designed to run in isolated environments:

```bash
# Install dependencies
pip install -r requirements.txt -r requirements-dev.txt

# Run tests with JUnit XML output (for CI)
pytest tests/ --junit-xml=test-results.xml --cov=core --cov-report=xml
```

## Writing New Tests

### Template

```python
import pytest
from core.your_module import YourClass

@pytest.fixture
def sample_data():
    """Create sample test data"""
    return {"key": "value"}

class TestYourClass:
    """Test YourClass functionality"""

    def test_initialization(self):
        """Test that class initializes correctly"""
        obj = YourClass()
        assert obj.attribute is not None

    def test_method_with_valid_input(self, sample_data):
        """Test method with valid input"""
        obj = YourClass()
        result = obj.method(sample_data)
        assert result == expected_value

    def test_method_raises_error_on_invalid_input(self):
        """Test that method raises error for invalid input"""
        obj = YourClass()
        with pytest.raises(ValueError):
            obj.method(invalid_input)
```

### Best Practices

1. **One assertion per test when possible** - Makes failures easier to diagnose
2. **Use descriptive test names** - `test_load_excel_with_missing_values` not `test_1`
3. **Test edge cases** - Empty data, None values, boundary conditions
4. **Use fixtures for reusable data** - Reduces code duplication
5. **Clean up resources** - Close files, matplotlib figures (`plt.close()`)
6. **Test both success and failure paths** - Valid inputs AND error conditions

## Troubleshooting

### Common Issues

**Import errors:**
```bash
# Ensure you're in the project root
cd /path/to/protein_stability
pytest tests/
```

**Module not found:**
```bash
# Install dependencies
pip install -r requirements.txt -r requirements-dev.txt
```

**Matplotlib backend errors:**
```python
# In test file, use non-interactive backend
import matplotlib
matplotlib.use('Agg')
```

**Pandas deprecation warnings:**
```bash
# Run with warnings as errors to catch issues
pytest tests/ -W error::DeprecationWarning
```

## Contributing

When adding new functionality:

1. Write tests FIRST (TDD approach recommended)
2. Aim for >80% coverage of new code
3. Run full test suite before committing: `pytest tests/`
4. Update this README if adding new test files

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
