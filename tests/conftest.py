"""Shared test fixtures"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import openpyxl
from openpyxl.styles import Font


@pytest.fixture
def sample_doe_data():
    """Create sample DoE data for testing"""
    return pd.DataFrame({
        'ID': [1, 2, 3, 4, 5],
        'Plate_96': ['Plate1', 'Plate1', 'Plate1', 'Plate1', 'Plate1'],
        'Well_96': ['A1', 'A2', 'A3', 'A4', 'A5'],
        'NaCl': [50, 100, 150, 50, 100],
        'pH': [6.0, 7.0, 8.0, 6.5, 7.5],
        'buffer': ['Tris', 'Tris', 'HEPES', 'Tris', 'HEPES'],
        'Response': [0.5, 0.7, 0.9, 0.6, 0.8]
    })


@pytest.fixture
def sample_doe_data_with_missing():
    """Create sample DoE data with missing values"""
    return pd.DataFrame({
        'ID': [1, 2, 3, 4, 5],
        'NaCl': [50, 100, np.nan, 50, 100],
        'pH': [6.0, 7.0, 8.0, np.nan, 7.5],
        'buffer': ['Tris', 'Tris', None, 'Tris', 'HEPES'],
        'Response': [0.5, 0.7, 0.9, np.nan, 0.8]
    })


@pytest.fixture
def temp_excel_file(sample_doe_data, tmp_path):
    """Create temporary Excel file with sample data"""
    filepath = tmp_path / "test_data.xlsx"
    sample_doe_data.to_excel(filepath, index=False)
    return filepath


@pytest.fixture
def temp_excel_with_stock_concs(sample_doe_data, tmp_path):
    """Create temporary Excel file with stock concentrations metadata"""
    filepath = tmp_path / "test_data_with_stocks.xlsx"

    # Create workbook with data sheet
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"

    # Write data
    for r_idx, row in enumerate([sample_doe_data.columns.tolist()] + sample_doe_data.values.tolist(), 1):
        for c_idx, value in enumerate(row, 1):
            ws.cell(row=r_idx, column=c_idx, value=value)

    # Create stock concentrations sheet
    stock_sheet = wb.create_sheet(title="Stock_Concentrations")
    stock_headers = ["Factor Name", "Stock Value", "Unit"]

    for col_idx, header in enumerate(stock_headers, start=1):
        cell = stock_sheet.cell(row=1, column=col_idx, value=header)
        cell.font = Font(bold=True)

    # Add stock concentration data
    stock_data = [
        ["NaCl (mM)", 1000, "mM"],
        ["Buffer pH", 14, "pH"]
    ]

    for row_idx, row_data in enumerate(stock_data, start=2):
        for col_idx, value in enumerate(row_data, start=1):
            stock_sheet.cell(row=row_idx, column=col_idx, value=value)

    wb.save(filepath)
    return filepath


@pytest.fixture
def sample_factorial_design():
    """Create sample full factorial design"""
    return pd.DataFrame({
        'Factor1': [10, 10, 20, 20],
        'Factor2': [100, 200, 100, 200],
        'Response': [0.5, 0.7, 0.6, 0.9]
    })


@pytest.fixture
def sample_analysis_results():
    """Create sample analysis results"""
    return {
        'model_type': 'main_effects',
        'aic': 45.2,
        'bic': 48.7,
        'r_squared': 0.85,
        'adj_r_squared': 0.82,
        'predictions': np.array([0.51, 0.69, 0.61, 0.88]),
        'residuals': np.array([-0.01, 0.01, -0.01, 0.02]),
        'formula': 'Response ~ Factor1 + Factor2'
    }
