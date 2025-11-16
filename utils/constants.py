"""
Shared constants for the DoE Suite
Consolidates constants from multiple modules
"""

# Available factors with display names (from designer_tab.py - most complete version)
AVAILABLE_FACTORS = {
    "buffer pH": "Buffer pH",
    "buffer_concentration": "Buffer Conc (mM)",
    "glycerol": "Glycerol (%)",
    "nacl": "NaCl (mM)",
    "kcl": "KCl (mM)",
    "zinc": "Zinc (mM)",
    "magnesium": "Magnesium (mM)",
    "calcium": "Calcium (mM)",
    "dmso": "DMSO (%)",
    "detergent": "Detergent",  # Categorical - accepts names like Tween-20, Triton, etc.
    "detergent_concentration": "Detergent (%)",
    "reducing_agent": "Reducing Agent",  # Categorical - accepts names like DTT, TCEP, BME, etc.
    "reducing_agent_concentration": "Reducing Agent (mM)",
}

# Metadata columns to exclude from analysis
METADATA_COLUMNS = ['ID', 'Plate_96', 'Well_96', 'Well_384', 'Source', 'Batch']
