#!/usr/bin/env python3
"""
Generate synthetic DoE data with real factor-response relationships
for testing multi-response analysis
"""
import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Define factor levels (factorial design)
# Use pH values that match BO's expected categorical values
buffer_ph_levels = [6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5]
buffer_conc_levels = [10, 50, 100]  # mM
nacl_levels = [0, 50, 150, 300]  # mM
zinc_levels = [0, 0.1, 1.0]  # mM
glycerol_levels = [0, 5, 10, 20]  # %

# Create full factorial design
import itertools
factorial_design = list(itertools.product(
    buffer_ph_levels,
    buffer_conc_levels,
    nacl_levels,
    zinc_levels,
    glycerol_levels
))

# Sample 96 runs (to fit a 96-well plate)
n_samples = 96
if len(factorial_design) > n_samples:
    # Use fractional factorial (random sample)
    indices = np.random.choice(len(factorial_design), n_samples, replace=False)
    selected_design = [factorial_design[i] for i in indices]
else:
    selected_design = factorial_design[:n_samples]

# Create DataFrame
data = pd.DataFrame(selected_design, columns=[
    'Buffer pH',
    'Buffer Conc (mM)',
    'NaCl (mM)',
    'Zinc (mM)',
    'Glycerol (%)'
])

# Add metadata columns
data['ID'] = range(1, len(data) + 1)
data['Plate_96'] = 1
data['Well_96'] = [f"{chr(65 + i//12)}{(i%12)+1}" for i in range(len(data))]


# ===== RESPONSE 1: Thermal Stability (Tm) =====
# Tm increases with:
# - Higher pH (optimal around 7-8)
# - Higher glycerol
# - Lower NaCl (too much salt destabilizes)
# - Zinc has small stabilizing effect

def calculate_tm(row):
    """Calculate Tm based on factor effects"""
    base_tm = 45.0  # Base melting temperature

    # pH effect (quadratic, optimal at pH 7.5)
    ph = row['Buffer pH']
    ph_effect = -0.5 * (ph - 7.5)**2 + 3.0

    # Glycerol effect (linear, stabilizes)
    glycerol_effect = 0.3 * row['Glycerol (%)']

    # NaCl effect (destabilizes at high concentration)
    nacl_effect = -0.01 * row['NaCl (mM)']

    # Zinc effect (small stabilization)
    zinc_effect = 0.5 * row['Zinc (mM)']

    # Buffer concentration (minimal effect)
    buffer_effect = 0.005 * row['Buffer Conc (mM)']

    # Interaction: pH and glycerol work together
    interaction_effect = 0.02 * (ph - 6.0) * row['Glycerol (%)']

    # Calculate Tm with some noise
    tm = base_tm + ph_effect + glycerol_effect + nacl_effect + zinc_effect + buffer_effect + interaction_effect

    # Add realistic noise
    noise = np.random.normal(0, 0.5)

    return tm + noise

data['Tm'] = data.apply(calculate_tm, axis=1)


# ===== RESPONSE 2: Aggregation (lower is better) =====
# Aggregation decreases with:
# - Higher buffer concentration
# - Lower pH (acidic conditions reduce aggregation for this protein)
# - NaCl helps prevent aggregation
# - Glycerol reduces aggregation

def calculate_aggregation(row):
    """Calculate aggregation percentage (0-100%)"""
    base_agg = 30.0  # Base aggregation

    # Buffer concentration effect (higher buffer reduces aggregation)
    buffer_effect = -0.15 * row['Buffer Conc (mM)']

    # pH effect (lower pH reduces aggregation)
    ph = row['Buffer pH']
    ph_effect = 2.0 * (ph - 6.0)

    # NaCl effect (moderate salt helps)
    nacl = row['NaCl (mM)']
    nacl_effect = -0.05 * nacl if nacl < 200 else -0.05 * 200 + 0.02 * (nacl - 200)

    # Glycerol effect (reduces aggregation)
    glycerol_effect = -0.4 * row['Glycerol (%)']

    # Zinc has no effect on aggregation

    # Calculate aggregation with noise
    agg = base_agg + buffer_effect + ph_effect + nacl_effect + glycerol_effect

    # Add noise
    noise = np.random.normal(0, 1.5)
    agg = agg + noise

    # Clamp to 0-100%
    return max(0, min(100, agg))

data['Aggregation'] = data.apply(calculate_aggregation, axis=1)


# ===== RESPONSE 3: Activity (higher is better) =====
# Activity increases with:
# - Optimal pH around 8.0
# - Zinc (cofactor)
# - Low NaCl
# - Moderate glycerol (too much inhibits)

def calculate_activity(row):
    """Calculate enzymatic activity (arbitrary units)"""
    base_activity = 50.0

    # pH effect (optimal at 8.0)
    ph = row['Buffer pH']
    ph_effect = -1.0 * (ph - 8.0)**2 + 5.0

    # Zinc effect (required cofactor)
    zinc_effect = 15.0 * row['Zinc (mM)']

    # NaCl effect (inhibits)
    nacl_effect = -0.03 * row['NaCl (mM)']

    # Glycerol effect (optimal at ~10%)
    glycerol = row['Glycerol (%)']
    glycerol_effect = -0.2 * (glycerol - 10)**2 + 2.0

    # Buffer concentration (minimal effect)
    buffer_effect = 0.01 * row['Buffer Conc (mM)']

    # Interaction: Zinc and pH
    interaction_effect = 2.0 * row['Zinc (mM)'] * (1 if ph > 7.0 else 0.5)

    # Calculate activity with noise
    activity = base_activity + ph_effect + zinc_effect + nacl_effect + glycerol_effect + buffer_effect + interaction_effect

    # Add noise
    noise = np.random.normal(0, 2.0)

    return max(0, activity + noise)

data['Activity'] = data.apply(calculate_activity, axis=1)


# Reorder columns (metadata, factors, responses)
column_order = [
    'ID', 'Plate_96', 'Well_96',  # Metadata
    'Buffer pH', 'Buffer Conc (mM)', 'NaCl (mM)', 'Zinc (mM)', 'Glycerol (%)',  # Factors
    'Tm', 'Aggregation', 'Activity'  # Responses
]
data = data[column_order]

# Save to Excel
output_file = 'test_multi_response_data.xlsx'
data.to_excel(output_file, index=False, sheet_name='DoE_Data')

print(f"✓ Generated {len(data)} DoE runs with synthetic data")
print(f"✓ Saved to: {output_file}")
print(f"\nFactor levels:")
print(f"  - Buffer pH: {buffer_ph_levels}")
print(f"  - Buffer Conc: {buffer_conc_levels} mM")
print(f"  - NaCl: {nacl_levels} mM")
print(f"  - Zinc: {zinc_levels} mM")
print(f"  - Glycerol: {glycerol_levels} %")
print(f"\nResponse variables:")
print(f"  - Tm (Thermal Stability): Higher is better, range ~{data['Tm'].min():.1f}-{data['Tm'].max():.1f} °C")
print(f"  - Aggregation: Lower is better, range ~{data['Aggregation'].min():.1f}-{data['Aggregation'].max():.1f} %")
print(f"  - Activity: Higher is better, range ~{data['Activity'].min():.1f}-{data['Activity'].max():.1f} AU")
print(f"\nExpected factor effects:")
print(f"  Tm: ↑ pH (7-8), ↑ Glycerol, ↓ NaCl, ↑ Zinc")
print(f"  Aggregation: ↑ Buffer Conc, ↓ pH, ↑ NaCl (moderate), ↑ Glycerol")
print(f"  Activity: ↑ pH (~8), ↑ Zinc (strong), ↓ NaCl, ↑ Glycerol (optimal ~10%)")
