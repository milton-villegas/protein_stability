"""Export service - generates Excel and CSV files in memory"""

import csv
import io
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

try:
    import openpyxl
    from openpyxl.styles import Font, Alignment, PatternFill
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False


def generate_excel_bytes(
    excel_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    stock_concs: Dict[str, float],
    project_name: str = "Design",
) -> bytes:
    """Generate Excel file as bytes with multiple sheets"""
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Sheet 1: Sample Tracking
        excel_df.to_excel(writer, sheet_name="Sample Tracking", index=False)

        # Sheet 2: Stock Concentrations
        if stock_concs:
            stock_df = pd.DataFrame([
                {"Factor": k, "Stock Concentration": v}
                for k, v in stock_concs.items()
            ])
            stock_df.to_excel(writer, sheet_name="Stock_Concentrations", index=False)

        # Sheet 3: Volume Data
        volume_df.to_excel(writer, sheet_name="Volumes", index=False)

    output.seek(0)
    return output.read()


def generate_csv_bytes(volume_df: pd.DataFrame) -> bytes:
    """Generate Opentrons-compatible CSV as bytes"""
    output = io.StringIO()
    volume_df.to_csv(output, index=False)
    return output.getvalue().encode("utf-8")
