"""
Data I/O utilities for CSV and Excel files
Shared between Designer and Analysis modules
"""
import csv
import pandas as pd
from typing import List, Dict, Optional, Tuple
from tkinter import filedialog

def load_csv(filepath: str) -> pd.DataFrame:
    """Load CSV file into DataFrame"""
    return pd.read_csv(filepath)

def load_excel(filepath: str, sheet_name: str = 0) -> pd.DataFrame:
    """Load Excel file into DataFrame"""
    return pd.read_excel(filepath, sheet_name=sheet_name)

def save_csv(filepath: str, data: pd.DataFrame, index: bool = False):
    """Save DataFrame to CSV"""
    data.to_csv(filepath, index=index)

def save_excel(filepath: str, data: pd.DataFrame, sheet_name: str = 'Sheet1', index: bool = False):
    """Save DataFrame to Excel"""
    data.to_excel(filepath, sheet_name=sheet_name, index=index)

def export_volumes_to_csv(filepath: str, headers: List[str], volume_rows: List[List]) -> None:
    """
    Export volume data to CSV for Opentrons

    Args:
        filepath: Output CSV path
        headers: Column headers
        volume_rows: List of rows with volume values
    """
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in volume_rows:
            writer.writerow(row)

def ask_open_file(title: str = "Select File", filetypes: Optional[List[Tuple[str, str]]] = None) -> Optional[str]:
    """
    Show file open dialog

    Args:
        title: Dialog title
        filetypes: List of (description, pattern) tuples, e.g., [("CSV Files", "*.csv")]

    Returns:
        Selected filepath or None if cancelled
    """
    if filetypes is None:
        filetypes = [("All Files", "*.*")]

    filepath = filedialog.askopenfilename(
        title=title,
        filetypes=filetypes
    )
    return filepath if filepath else None

def ask_save_file(title: str = "Save File", filetypes: Optional[List[Tuple[str, str]]] = None,
                  defaultextension: str = "") -> Optional[str]:
    """
    Show file save dialog

    Args:
        title: Dialog title
        filetypes: List of (description, pattern) tuples
        defaultextension: Default file extension (e.g., ".csv")

    Returns:
        Selected filepath or None if cancelled
    """
    if filetypes is None:
        filetypes = [("All Files", "*.*")]

    filepath = filedialog.asksaveasfilename(
        title=title,
        filetypes=filetypes,
        defaultextension=defaultextension
    )
    return filepath if filepath else None
